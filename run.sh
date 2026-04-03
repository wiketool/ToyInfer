./build/ToyInfer --model_dir=models/Qwen3-4B \
  --max_seq_len=40960 --thinking 1

./build/ToyInfer --model_dir=models/Qwen3-4B \
  --max_seq_len=40960 --detail_time=1 \
  --use_dedicated_prefill=1 \
  --use_multi_stream=1 \
  --bench=short

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --output=toyinfer_nsys \
  ./build/ToyInfer --model_dir=models/Qwen3-4B \
  --max_seq_len=40960 --thinking 1

ncu --set basic -o ncu_basic \
  ./build/ToyInfer --model_dir=models/Qwen3-4B \
  --max_seq_len=40960 --detail_time=1 \
  --use_dedicated_prefill=1 \
  --use_multi_stream=1 \
  --bench=short

# stream关闭，graph关闭，bench=long
./build/ToyInfer --model_dir=models/Qwen3-4B   \
  --max_seq_len=40960 --detail_time=1 \
  --use_dedicated_prefill=1 \
  --use_multi_stream=0 \
  --enable_cuda_graph=0 \
  --bench=long

# stream关闭，graph关闭，bench=long
./build/ToyInfer --model_dir=models/Qwen3-4B   --max_seq_len=40960 --detail_time=1   --use_dedicated_prefill=1   --use_multi_stream=0   --enable_cuda_graph=0 --bench=long

# stream关闭，graph关闭，bench=long
./build/ToyInfer --model_dir=models/Qwen3-4B   --max_seq_len=40960 --detail_time=1   --use_dedicated_prefill=1   --use_multi_stream=0   --enable_cuda_graph=0 --bench=long