# 一些碎碎念

## Kernel后续优化

- [] softmax_f32_kernel分片处理，使用online softmax算法，减少和global memory交互的次数
- [] Flash attention 解决prefill时softmax显存爆炸和memory bottleneck的情况
- [] cudaStream 优化
- [] cudaGraph 优化
