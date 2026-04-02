./build/ToyInfer --model_dir=models/Qwen3-4B \
  --max_seq_len=40960 --thinking 1

./build/ToyInfer --model_dir=models/Qwen3-4B --max_seq_len=40960 --detail_time=1 --use_dedicated_prefill=1 --use_multi_stream=1 --enable_cuda_graph=1
