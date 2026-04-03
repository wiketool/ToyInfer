LOG=ncu.log
CMD="./build/ToyInfer --model_dir=models/Qwen3-4B \
  --max_seq_len=40960 --detail_time=1 \
  --use_dedicated_prefill=1 \
  --use_multi_stream=1 \
  --enable_cuda_graph=0 \
  --bench=long"

echo "" | tee -a $LOG
echo "===== PD两阶段，多流，关闭CUDA Graph，bench=long | NCU基础采样 =====" | tee -a $LOG
echo "ncu --kill yes --launch-count 30 --target-processes all --set default -o ncu_basic_pd1_st1_cg0_long \$CMD" | tee -a $LOG
ncu --kill yes --launch-count 30 --target-processes all --set default -o ncu_basic_pd1_st1_cg0_long $CMD 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "===== PD两阶段，多流，关闭CUDA Graph，bench=long | NCU-FLOPS指标 =====" | tee -a $LOG
echo "ncu --kill yes --launch-count 30 --target-processes all \
  --metrics sm__cycles_elapsed.avg,smsp__cycles_elapsed.avg,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__inst_executed_pipe_tensor.sum \
  -o ncu_flops_pd1_st1_cg0_long \$CMD" | tee -a $LOG

ncu --kill yes --launch-count 30 --target-processes all \
  --metrics sm__cycles_elapsed.avg,smsp__cycles_elapsed.avg,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__inst_executed_pipe_tensor.sum \
  -o ncu_flops_pd1_st1_cg0_long $CMD 2>&1 | tee -a $LOG

LOG=ncu.log
CMD="./build/ToyInfer --model_dir=models/Qwen3-4B \
  --max_seq_len=40960 --detail_time=1 \
  --use_dedicated_prefill=0 \
  --use_multi_stream=1 \
  --enable_cuda_graph=0 \
  --bench=long"

echo "" | tee -a $LOG
echo "===== prefill走docode，多流，关闭CUDA Graph，bench=long | NCU基础采样 =====" | tee -a $LOG
echo "ncu --kill yes --launch-count 30 --target-processes all --set default -o ncu_basic_pd0_st1_cg0_long \$CMD" | tee -a $LOG
ncu --kill yes --launch-count 30 --target-processes all --set default -o ncu_basic_pd0_st1_cg0_long $CMD 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "===== prefill走docode，多流，关闭CUDA Graph，bench=long | NCU-FLOPS指标 =====" | tee -a $LOG
echo "ncu --kill yes --launch-count 30 --target-processes all \
  --metrics sm__cycles_elapsed.avg,smsp__cycles_elapsed.avg,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__inst_executed_pipe_tensor.sum \
  -o ncu_flops_pd0_st1_cg0_long \$CMD" | tee -a $LOG

ncu --kill yes --launch-count 30 --target-processes all \
  --metrics sm__cycles_elapsed.avg,smsp__cycles_elapsed.avg,gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__inst_executed_pipe_tensor.sum \
  -o ncu_flops_pd0_st1_cg0_long $CMD 2>&1 | tee -a $LOG