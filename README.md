# ReActNet_FPGA
ReActNet FPGA Accelerator

## Introduction
基于Vitis HLS实现一个ReActNet加速器
代码注释还待详细

## Network Architecture
参考论文——ReActNet: Towards Precise Binary Neural Network with Generalized Activation Functions
第一层（输入层）和最后一层（分类器）保持全精度，其余为二值化卷积。
骨干网络为ResNet20

## Document Structure
上传网络结构文件和权重文件。（权重不是训练结果，是暂时乱写的，主要验证加速器设计）
|---src

  |-react.h（主网络头文件）
  
  |-react.cpp（主网络结构）
  
  |-conv_weight.h（卷积权重）
  
  |-dimension_def.h（维度定义）
  
  |-layer.h（功能层定义：batchnorm、量化、shortcut、...）
  
  |-conv.h（卷积定义）
  
  |-typedefs.h（数据类型定义）
  
  |-weight_fp.h（全精度权重）


## Synthesis Result
| Resource   | Value   |
|-------|-------|
| DSP | 122 |
| BRAM | 465 |
| LUT | 30228 |
| FPS | 750 |
| POWER | 3.0 |

## Reference
https://github.com/cornell-zhang/FracBNN
