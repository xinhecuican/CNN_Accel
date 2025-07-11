教学用CNN加速器，可以加速卷积及池化操作

# 特性

- 基于LaCC接口, 适配[OpenLA500](https://gitee.com/loongson-edu/open-la500)。
- 卷积操作使用脉动阵列实现，由硬件实现Im2col数据重排，支持ReLu激活函数。
- 参数高度可配置，卷积核大小，步幅，缓存大小，填充都可以使用指令配置，执行int8, int16, int32的参数及权重，可以通过划分实现通用的卷积计算。
- 支持卷积数据与缓存中数据相加再写回，支持结果与偏置相加。

[指令文档可见](doc/指令.md)

[功能文档可见](doc/功能.md)

quant lenet could see: https://github.com/xinhecuican/lenet_int8