# 一些碎碎念

## Kernel后续优化

- [ ] softmax_f32_kernel分片处理，使用online softmax算法，减少和global memory交互的次数
- [ ] Flash attention 解决prefill时softmax显存爆炸和memory bottleneck的情况
- [x] cudaStream 优化
- [x] cudaGraph 优化
- [x] logits 变为pinned memory，减少copy开销
- [ ] 测量flash attention中对于计算O的时候，到底是存score划算还是重新算score划算；现在用的是存score
