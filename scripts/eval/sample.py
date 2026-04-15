import os
import json
import argparse
import random
import logging
from tqdm import tqdm
import concurrent.futures
from typing import List, Dict, Set, Tuple
from openai import OpenAI
from transformers import AutoTokenizer
import requests

import sys
sys.path.append("DeepSeek-V3.2/hf_model/encoding")

from encoding_dsv32 import encode_messages

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process data in parallel")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to run simultaneously")
    parser.add_argument("--remote_url", type=str, nargs='+', default=None, help="Remote URL for Ray")
    parser.add_argument("--input_file", type=str,    required=True, help="Path to input JSON or JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--sample", type=int, default=-1, help="Number of samples to process (-1 for all)")
    parser.add_argument("--generate_max_length", type=int, default=100, help="Max length for generation")
    parser.add_argument("--tokenizer_name", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--use_custom_template", type=str, default="no", help="Model name")
    parser.add_argument("--n", type=int, default=16, help="Number of samples to generate")
    return parser.parse_args()

def load_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        if file_path.endswith('.json'):
            return json.load(f)
        elif file_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
            for line in data:
                line["id"] = line["extra_info"]["index"]
            return data
        elif file_path.endswith('.parquet'):
            import pandas as pd
            df = pd.read_parquet(file_path)
            data = df.to_dict(orient="records")
            for line in data:
                line["id"] = line["extra_info"]["index"]
            return data
        else:
            raise ValueError("Input file must be either JSON or JSONL")

def load_unprocessed_data(data: List[Dict], output_dir: str) -> Set[int]:
    if not os.path.exists(output_dir):
        return set(range(len(data)))
    
    processed_ids = set()
    if output_dir.endswith(".jsonl"):
        with open(output_dir, "r") as f:
            for line in f:
                try:
                    idx = json.loads(line)["id"]
                    processed_ids.add(idx)
                except:
                    pass
    else:
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".jsonl"):
                with open(os.path.join(output_dir, file_name), "r") as f:
                    for line in f:
                        try:
                            idx = json.loads(line)["id"]
                            processed_ids.add(idx)
                        except:
                            pass
                    
    return set(range(len(data))) - set(idx for idx, item in enumerate(data) if str(item["id"]) in processed_ids)




def process_data_thread(
    args: argparse.Namespace, 
    data_index: int, 
    data: List[Dict], 
    tokenizer: AutoTokenizer,
    process_idx: int, 
) -> Dict:
    output_file_path = os.path.join(args.output_dir, f"output_subprocess_{process_idx}.jsonl") if not args.output_dir.endswith(".jsonl") else args.output_dir
    
    item = data[data_index]
    if args.use_custom_template == "no":
        prompt1 = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"][0]["content"]}],
                tokenize=False, 
                add_generation_prompt=True
            )
    else:
        # 这里thinking_mode 可选 thinking/chat
        encode_config = dict(thinking_mode="thinking", drop_thinking=True, add_default_bos_token=True)

        prompt1 = encode_messages([{"role": "user", "content": item["prompt"][0]["content"]}], **encode_config)
        

    prompt1 = prompt1.removeprefix("<｜begin▁of▁sentence｜>")
    item["src_prompt"] = prompt1
    url = random.choice(args.remote_url)

    data = {
        "model": "qwen3-4b", # 模型路径
        "prompt": prompt1,
        "temperature": args.temperature,
        "max_tokens": args.generate_max_length,
        "n": args.n,
        "skip_special_tokens": False,
    }

    response = requests.post(
        url+"/completions",
        json=data
    )
    ret = response.json()


    for choice in ret["choices"]:
        item["pred"]=choice["text"]
        item["stop_reason"] = choice["finish_reason"]
        with open(output_file_path, "a+") as output_file:
            output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            output_file.flush()




def main():
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    logger.info(f"Arguments: {args}")

    # assert args.n == 1, "n must be 1, n>1 haven't verify resume"
    # Create output directory
    # os.makedirs(args.output_dir, exist_ok=True)
    print(args.output_dir)
    # Load data
    data = load_data(args.input_file)
    
    if args.sample > 0:
        data = data[:min(len(data), args.sample)]
    total_samples = len(data)
    print("total_samples",total_samples)
    num_threads = args.num_processes

    unprocessed_indices = load_unprocessed_data(data, args.output_dir)
    remaining_indices = list(unprocessed_indices)

    # 使用 ThreadPoolExecutor 进行并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(
                process_data_thread, 
                args, 
                data_index,  # 只传递索引
                data, 
                tokenizer,
                idx%num_threads, 
            ): data_index
            for idx, data_index in enumerate(remaining_indices)
        }
        
        logger.info("Waiting for all threads to finish...")
        
        # 使用 tqdm 包装 as_completed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures),desc="Processing"):
            result = future.result()

    logger.info("All threads have finished processing")

if __name__ == "__main__":
    main()
