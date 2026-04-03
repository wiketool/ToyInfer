LOG=nsys.log
CMD="./build/ToyInfer --model_dir=models/Qwen3-4B \
  --max_seq_len=40960 --detail_time=1 \
  --use_dedicated_prefill=1 \
  --use_multi_stream=1 \
  --enable_cuda_graph=0 \
  --bench=long"


echo "" | tee -a $LOG
echo "===== PD两阶段，多流，关闭CUDA Graph，bench=long | NSYS+NVTX =====" | tee -a $LOG
echo "nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0  \
  --capture-range=nvtx --capture-range-end=stop -p engine.chat.turn \
  --output=nsys_pd1_st1_cg0_long_nvtx \$CMD" | tee -a $LOG

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
  --capture-range=nvtx \
  --capture-range-end=stop \
  -p engine.chat.turn \
  --output=nsys_pd1_st1_cg0_long_nvtx \
  ./build/ToyInfer --model_dir=models/Qwen3-4B \
  --max_seq_len=40960 --detail_time=1 \
  --use_dedicated_prefill=1 \
  --use_multi_stream=1 \
  --enable_cuda_graph=0 \
  --bench=long


nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
  --capture-range=nvtx --capture-range-end=stop -p engine.chat.turn \
  --output=nsys_pd1_st1_cg0_long_nvtx $CMD 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "===== 导出 NSYS SQLite =====" | tee -a $LOG
echo "nsys export --type sqlite --output nsys_pd1_st1_cg0_long_nvtx nsys_pd1_st1_cg0_long_nvtx.nsys-rep" | tee -a $LOG
nsys export --type sqlite --output nsys_pd1_st1_cg0_long_nvtx nsys_pd1_st1_cg0_long_nvtx.nsys-rep 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "===== PD两阶段，多流，关闭CUDA Graph，bench=long | NSYS全程采样（无NVTX兜底） =====" | tee -a $LOG
echo "nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output=nsys_pd1_st1_cg0_long_full \$CMD" | tee -a $LOG

nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --output=nsys_pd1_st1_cg0_long_full $CMD 2>&1 | tee -a $LOG