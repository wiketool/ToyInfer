# export.py

import argparse
import gzip
import json
import math
import os
import shutil
import struct
from pathlib import Path

import json
from jinja2 import Template


# BBPE训练过程中使用了一个特殊的字节到Unicode字符的映射，以确保所有字节都能安全地表示为Unicode字符。
# 这是因为某些字节可能在UTF-8编码中被解释为多字节序列的一部分，导致解析错误。
# 通过将这些不安全的字节映射到256开始的Unicode字符上，可以确保每个字节都有一个唯一且安全的Unicode表示，从而避免了编码问题。
# 如果不这样做，在训练过程中可能会遇到正则表达式错误，或者vocab无法写入json的问题。

# 将0-255的字节映射到可见的Unicode字符上，确保所有字节都能安全地表示为Unicode字符
def bytes_to_unicode():
    """Reference GPT-2 byte→Unicode map."""
    # ASCII中可见字符【0-127】
    bs = list(range(ord("!"), ord("~") + 1))
    # 扩展拉丁【128-255】中可见字符,跳过173(软连字符)
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        # 对于不在bs中的字节，即为不安全字符，将其映射到256开始的unicode字符上
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

def internal_to_bytes(U2B, token_str: str) -> bytes:
    return b''.join(
        bytes([U2B[ch]]) if ch in U2B else ch.encode('utf-8')
        for ch in token_str
    )

def build_tokenizer(model, output_dir):
    B2U = bytes_to_unicode()
    U2B = {u: b for b, u in B2U.items()}

    tokenizer = model.tokenizer

    # Get ID → token mapping
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    all_tokens = [id_to_token[i] for i in sorted(id_to_token)]

    tokenizer_data = json.loads(tokenizer.backend_tokenizer.to_str())

    # Extract vocab and merge rules
    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"]["merges"]

    # Build merge rank table
    merge_rank = {''.join(tuple(merge if isinstance(merge, list) else merge.split())): i for i, merge in enumerate(merges)}

    # Create pseudo-score dictionary
    # Tokens from initial vocab get score 0 (unmerged tokens)
    # Merged tokens get scores based on merge rank
    pseudo_scores = {}
    for token_id, token in enumerate(all_tokens):
        # If this token was the result of a merge, it will appear in merge_rank
        rank = merge_rank.get(token)

        if rank is not None:
            score = -math.log(rank + 1)
        else:
            score = -1e6  # Initial vocab tokens
        pseudo_scores[token] = score

    max_token_length = max(len(t) for t in all_tokens)
    tokenizer_path = os.path.join(output_dir, "tokenizer.bin")

    with open(tokenizer_path, "wb") as out_f:
        out_f.write(struct.pack("<I", len(all_tokens)))       # 4 bytes: number of tokens

        for id, token in enumerate(all_tokens):
            token_bytes = internal_to_bytes(U2B, token)
            out_f.write(struct.pack("f", pseudo_scores[token])) # 4 bytes: merge score
            out_f.write(struct.pack("<I", len(token_bytes))) # 4 bytes: token length
            out_f.write(token_bytes)                         # UTF-8 bytes

    print(f"Written tokenizer model to {tokenizer_path}")

def build_prompts(model, output_dir):
    template = Template(model.tokenizer.chat_template)

    # Template 1: User
    messages = [{"role": "user", "content": "%s"}]
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    with open(os.path.join(output_dir, 'template_user.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 2: User with Thinking
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
    with open(os.path.join(output_dir, 'template_user_thinking.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 3: System + User
    messages = [{"role": "system", "content": "%s"}, {"role": "user", "content": "%s"}]
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
    with open(os.path.join(output_dir, 'template_system.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    # Template 4: System + User with Thinking
    rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
    with open(os.path.join(output_dir, 'template_system_thinking.txt'), 'w', encoding='utf-8', newline='') as f:
        f.write(rendered_prompt)

    print(f"Written prompt templates to '{output_dir}'")

# -----------------------------------------------------------------------------
# Load / import functions

def load_tokenizer_and_config(model_path):
    """Loads only the tokenizer and config, not the full model weights."""
    try:
        from transformers import AutoConfig, AutoTokenizer
        from types import SimpleNamespace
    except ImportError:
        print("Error: transformers package is required.")
        print("Please run `pip install transformers` to install it.")
        return None

    print(f"Loading tokenizer and config from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf_config = AutoConfig.from_pretrained(model_path)

    model_mock = SimpleNamespace()
    model_mock.tokenizer = tokenizer
    model_mock.bos_token_id = hf_config.bos_token_id if hasattr(hf_config, "bos_token_id") else 0
    model_mock.eos_token_id = hf_config.eos_token_id if hasattr(hf_config, "eos_token_id") else 0
    
    print("Successfully loaded tokenizer and config.")
    return model_mock

def read_safetensor_header(filepath):
    """
    原生的读取 safetensors 头部数据，不依赖外部库。
    """
    with open(filepath, 'rb') as f:
        # 读取前 8 字节，这是一个无符号的 64 位整型（小端序），代表 JSON 头部的长度
        header_len_bytes = f.read(8)
        if not header_len_bytes:
            return {}
        header_len = struct.unpack('<Q', header_len_bytes)[0]
        
        # 读取这 N 个字节并解析为 JSON
        header_json_bytes = f.read(header_len)
        header = json.loads(header_json_bytes.decode('utf-8'))
        
    # safetensors 通常自带一个 __metadata__ key，我们不需要它
    if '__metadata__' in header:
        del header['__metadata__']
        
    return header

def export_safetensors_config(model_dir: str, output_filepath="safetensors_config.json"):
    """
    接受模型文件夹路径，扫描并导出详细的张量信息。
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    config_dict = {}
    index_file = model_path / "model.safetensors.index.json"

    # 情况 A: 分片模型 (存在 index.json)
    if index_file.exists():
        print(f"[INFO] 发现分片索引文件: {index_file.name}")
        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get("weight_map", {})
        
        # 将张量先按所在的文件分组，避免同一个文件被重复打开多次
        files_to_tensors = {}
        for tensor_name, filename in weight_map.items():
            files_to_tensors.setdefault(filename, []).append(tensor_name)
            
        file_base_offset = 0
        # 按照文件名排序，确保按照固定的文件顺序拼接
        for filename in sorted(files_to_tensors.keys()):
            filepath = model_path / filename
            if not filepath.exists():
                print(f"[WARN] 权重文件不存在跳过: {filename}")
                continue
                
            header = read_safetensor_header(filepath)
            
            max_offset = 0
            for tensor_name in files_to_tensors[filename]:
                if tensor_name in header:
                    info = header[tensor_name]
                    config_dict[tensor_name] = {
                        "name": tensor_name,
                        "offset": file_base_offset + info["data_offsets"][0],
                        "size": info["data_offsets"][1] - info["data_offsets"][0]
                    }
                    if info["data_offsets"][1] > max_offset:
                        max_offset = info["data_offsets"][1]
            # 累加当前文件的数据块长度
            file_base_offset += max_offset

    # 情况 B: 单体模型 (只存在普通 .safetensors 文件)
    else:
        # 按名称排序避免不同平台读取顺序带来的差异
        safetensor_files = sorted(list(model_path.glob("*.safetensors")))
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors found in {model_dir}")
        
        print(f"[INFO] 发现 {len(safetensor_files)} 个 safetensors 文件")
        file_base_offset = 0
        for file in safetensor_files:
            header = read_safetensor_header(file)
            filename = file.name
            
            max_offset = 0
            for tensor_name, info in header.items():
                config_dict[tensor_name] = {
                    "name": tensor_name,
                    "offset": file_base_offset + info["data_offsets"][0],
                    "size": info["data_offsets"][1] - info["data_offsets"][0]
                }
                if info["data_offsets"][1] > max_offset:
                    max_offset = info["data_offsets"][1]
            # 累加当前文件的数据块长度
            file_base_offset += max_offset
                
    # 按照计算出的全局递增 offset 排序
    sorted_config = sorted(config_dict.values(), key=lambda x: x["name"])
    
    # 写入用于 C++ 构造初始化的代码片段
    out_path = Path(output_filepath).with_suffix(".inc")
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in sorted_config:
            f.write(f'    {{"{item["name"]}", {item["offset"]}, {item["size"]}}},\n')
            
    print(f"[SUCCESS] 成功提取了 {len(sorted_config)} 个权重的位置信息！")
    print(f"[SUCCESS] C++ 初始化配置已导出至: {out_path.absolute()}")
    
    return sorted_config



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tokenizer.bin and template files")
    parser.add_argument("model_path", type=str, help="Path to the local Hugging Face model directory (used for both input and output).")
    args = parser.parse_args()

    model_info = load_tokenizer_and_config(args.model_path)

    if model_info:
        build_tokenizer(model_info, args.model_path)
        build_prompts(model_info, args.model_path)
        export_safetensors_config(args.model_path, os.path.join(args.model_path, "safetensors_config.json"))
