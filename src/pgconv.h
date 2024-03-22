#ifndef PGCONV_H
#define PGCONV_H

#include "typedefs.h"
#include "dimension_def.h"
#include <iostream>

using namespace std;
const uint64 m1 = 6148914691236517205;
const uint64 m2 = 3689348814741910323;
const uint64 m4 = 1085102592571150095;

inline uint8 compute_engine_64(uint64 b, uint64 w)
{
#pragma HLS latency max=1
    uint64 x = b^w;

    x -= (x >> 1) & m1;
    x = (x & m2) + ((x >> 2) & m2);
    x = (x + (x >> 4)) & m4;
    x += x >>  8;
    x += x >> 16;
    x += x >> 32;
    return (x & 0x7f);
}

//inline uint8 compute_engine_64(uint64 b, uint64 w)
//{
//#pragma HLS latency max=1
//	uint64 t = b^w;
//	uint8 sum = 0;
//	for(int i = 0; i < 64; i++){
//#pragma HLS UNROLL
//		sum += t[i];
//	}
//	// use yichi method
//	return sum;
//}


/*
 * Attention:
 * when lsb_outputs is not used, for example, in the first binary conv layer,
 * its values are still modified.
 * This makes the next accumulation in the lsb_outputs buffer incorrect because
 * the registers directly copy the values from the buffer.
 */

/*
 * Binary convolutional layer
 */
void binary_conv3x3_tile(
		uint64 inputs[WIDTH][WIDTH],
		const uint64 weights[OUT_CHANNEL_PARALLELISM][3][3],
		int16 outputs[CHANNEL_OUT_T][WIDTH][WIDTH],

		int c_in,
		int in_channels,
		int H_fmap_in,
		int stride,
		int padding
)
{
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=weights complete dim=3
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=1
	uint64 line_buffer[2][WIDTH] = {0};
	uint64 window_buffer[3][3] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=0

	int16 partial_out_feature[OUT_CHANNEL_PARALLELISM] = {0};
#pragma HLS ARRAY_PARTITION variable=partial_out_feature complete dim=1

	Loop_Tile:
	for (int row = 0; row < H_fmap_in+1; row++)
	{
		for (int col = 0; col < H_fmap_in+1; col++)
		{
#pragma HLS PIPELINE
			// update window buffer and line buffer
			for (int i=0; i<3; i++) {
				window_buffer[i][0] = window_buffer[i][1];
				window_buffer[i][1] = window_buffer[i][2];
			}

			window_buffer[0][2] = (line_buffer[0][col]);
			window_buffer[1][2] = (line_buffer[0][col] = line_buffer[1][col]);
			window_buffer[2][2] = (line_buffer[1][col] = line_buffer[row][col]);

			if (row + padding >= 2 && col + padding >= 2 && row % stride == 0 && col % stride == 0)
			{
				// copy output features into registers
				for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
					if (c_in > 0){
						partial_out_feature[channel_pt] = outputs[channel_pt][row][col];
					}
					else{
						partial_out_feature[channel_pt] = 0;
					}
				}

				// Compute each feature in an output channel
				for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
					int16 accumulation = 0;
					// Compute each output channel
					for (int k_row=0; k_row<3; k_row++) {
						for (int k_col=0; k_col<3; k_col++) {
							int row_idx_pad = row - k_row;
							int col_idx_pad = col - k_col;
							if(row_idx_pad>=0 && row_idx_pad<H_fmap_in && col_idx_pad>=0 && col_idx_pad<H_fmap_in){
								uint64 a = window_buffer[2-k_row][2-k_col];
								uint64 w = weights[channel_pt][2-k_row][2-k_col];
								accumulation += in_channels - 2*compute_engine_64(a, w);
							}
						}
					}

					partial_out_feature[channel_pt] += accumulation;
				}

				for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
					int out_row = (row-(2-padding)) / stride;
					int out_col = (col-(2-padding)) / stride;
					outputs[channel_pt][out_row][out_col] = partial_out_feature[channel_pt];
				}
			}
		}
	}

	return;
}

/*
 * full-precision convolutional layer
 */
void fp_conv3x3_tile(
		FIX_FM_acc inputs[WIDTH][WIDTH],
		const FIX_FM_acc weights[OUT_CHANNEL_PARALLELISM][3][3],
		FIX_FM_acc outputs[CHANNEL_OUT_T][WIDTH][WIDTH],

		int c_in,
		int in_channels,
		int H_fmap_in,
		int stride,
		int padding
)
{
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=weights complete dim=3
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=1
	FIX_32_10 line_buffer[2][WIDTH] = {0};
    FIX_32_10 window_buffer[3][3] = {0};
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=0
#pragma HLS ARRAY_PARTITION variable=window_buffer complete dim=0

    FIX_FM_acc partial_out_feature[OUT_CHANNEL_PARALLELISM] = {0};
#pragma HLS ARRAY_PARTITION variable=partial_out_feature complete dim=1

	Loop_Tile:
	for (int row = 0; row < H_fmap_in+1; row++)
	{
		for (int col = 0; col < H_fmap_in+1; col++)
		{
#pragma HLS PIPELINE
			// update window buffer and line buffer
			for (int i=0; i<3; i++) {
				window_buffer[i][0] = window_buffer[i][1];
				window_buffer[i][1] = window_buffer[i][2];
			}

			window_buffer[0][2] = (line_buffer[0][col]);
			window_buffer[1][2] = (line_buffer[0][col] = line_buffer[1][col]);
			window_buffer[2][2] = (line_buffer[1][col] = line_buffer[row][col]);

			if (row + padding >= 2 && col + padding >= 2 && row % stride == 0 && col % stride == 0)
			{
				// copy output features into registers
				for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
					if (c_in > 0){
						partial_out_feature[channel_pt] = outputs[channel_pt][row][col];
					}
					else{
						partial_out_feature[channel_pt] = 0;
					}
				}

				// Compute each feature in an output channel
				for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
					FIX_FM_acc accumulation = 0;
					// Compute each output channel
					for (int k_row=0; k_row<3; k_row++) {
						for (int k_col=0; k_col<3; k_col++) {
							int row_idx_pad = row - k_row;
							int col_idx_pad = col - k_col;
							if(row_idx_pad>=0 && row_idx_pad<H_fmap_in && col_idx_pad>=0 && col_idx_pad<H_fmap_in){
								uint64 a = window_buffer[2-k_row][2-k_col];
								uint64 w = weights[channel_pt][2-k_row][2-k_col];
								accumulation += a * w;
							}
						}
					}

					partial_out_feature[channel_pt] += accumulation;
				}

				for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
					outputs[channel_pt][row-(2-padding)][col-(2-padding)] = partial_out_feature[channel_pt];
				}
			}
		}
	}

	return;
}

void fp_conv1x1_tile(
		FIX_FM_acc inputs[WIDTH][WIDTH],
		const FIX_FM_acc weights[OUT_CHANNEL_PARALLELISM],
		FIX_FM_acc outputs[CHANNEL_OUT_T][WIDTH][WIDTH],

		int c_in,
		int in_channels,
		int H_fmap_in,
		int stride,
		int padding
)
{
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=weights complete dim=3
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=1

	FIX_FM_acc partial_out_feature[OUT_CHANNEL_PARALLELISM] = {0};
#pragma HLS ARRAY_PARTITION variable=partial_out_feature complete dim=1

    Loop_Tile:
	for (int row = 0; row < H_fmap_in+1; row++)
	{
		for (int col = 0; col < H_fmap_in+1; col++)
		{
#pragma HLS PIPELINE
			// copy output features into registers
			for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
				if (c_in > 0){
					partial_out_feature[channel_pt] = outputs[channel_pt][row][col];
				}
				else{
					partial_out_feature[channel_pt] = 0;
				}
			}

			// Compute each feature in an output channel
			for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
				FIX_FM_acc accumulation = 0;
				// Compute each output channel
				FIX_FM_acc a = inputs[row][col];
				FIX_FM_acc w = weights[channel_pt];
				accumulation += a * w;

				partial_out_feature[channel_pt] += accumulation;
			}

			for (int channel_pt=0; channel_pt<OUT_CHANNEL_PARALLELISM; channel_pt++) {
				outputs[channel_pt][row][col] = partial_out_feature[channel_pt];
			}
		}
	}

	return;
}

#endif

