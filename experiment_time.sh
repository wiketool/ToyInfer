LOG=run.log
echo "===== START $(date) =====" > $LOG

for bench in long short; do
  for prefill in 0 1; do
    for stream in 0 1; do
      for graph in 0 1; do

        # 生成中文描述
        if [ $prefill -eq 0 ]; then pd="PD不分离"; else pd="PD分离"; fi
        if [ $stream -eq 0 ]; then st="单流"; else st="多流"; fi
        if [ $graph -eq 0 ]; then cg="关闭CUDA Graph"; else cg="开启CUDA Graph"; fi

        desc="$pd，$st，$cg，bench=$bench"

        echo "" | tee -a $LOG
        echo "===== $desc =====" | tee -a $LOG

        cmd="./build/ToyInfer --model_dir=models/Qwen3-4B \
          --max_seq_len=40960 --detail_time=1 \
          --use_dedicated_prefill=$prefill \
          --use_multi_stream=$stream \
          --enable_cuda_graph=$graph \
          --bench=$bench"

        # 打印命令
        echo "$cmd" | tee -a $LOG

        # 执行并记录 stdout + stderr
        eval $cmd 2>&1 | tee -a $LOG

      done
    done
  done
done

echo "===== END $(date) =====" | tee -a $LOG