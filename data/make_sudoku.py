import reasoning_gym
from datasets import Dataset
import json
import os
from tqdm import tqdm

# ================= 配置路径 =================
# 你原来的路径 (取消注释并修改即可)
base_path = "./reasoning-unk/data"
# base_path = "./data"  # 临时本地路径，方便测试
os.makedirs(base_path, exist_ok=True)

prefix = "unk"
filename_suffix = "sudoku_100"

# ================= Prompt 模板 =================
# 针对数独适配了你的 "显式 Unknown" 风格模板
TEMPLATE = """Below is a Sudoku puzzle represented as a single string of 81 digits (row by row).
* `_` denotes an empty cell.
* `1-9` denote filled cells.

**Input Puzzle:**
{puzzle}


**Rules:**
1. You must output **only** the completed grid as 9 rows of 9 digits.
2. Do NOT use \\boxed{{...}}, markdown code blocks (```), or any other formatting.
3. Do NOT include any introductory text, explanations, or labels. Just the numbers.
4. Separate each row with a newline.

Example format:
4 8 3 9 2 1 6 5 7
9 1 2 3 4 5 6 7 8
...

Important:
If, after several reasonable attempts, you believe you cannot find a correct solution (e.g. the puzzle is invalid), do NOT output \\boxed{{...}}. Reply in plain text starting with:
"Sorry, I can't solve this problem. Here is how far I got: ..." and then briefly summarize your reasoning.
"""

def generate_sudoku_eval_set():
    print("正在生成 Sudoku 数据集 (Size=100, All Eval)...")

    # 生成 100 条数据
    # reasoning_gym 的 sudoku 默认生成 9x9 标准数独
    dataset = reasoning_gym.create_dataset(
        'sudoku',
        size=100,
        seed=42,
        min_empty=30,
        max_empty=50
    )
    eval_dataset = []

    for i, data in enumerate(tqdm(dataset)):
        # 提取题目和答案
        # reasoning_gym sudoku 通常返回:
        # question: "003020600900305001..." (81 chars)
        # answer:   "483921657912345871..." (81 chars)
        puzzle_str = data['question']
        solution_str = data['answer']
        
        # 构造 prompt
        prompt_content = TEMPLATE.format(puzzle=puzzle_str)

        sample = {
            "data_source": "reasoning-gym-sudoku",
            "prompt": [{
                "role": "user",
                "content": prompt_content
            }],
            "ability": "reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({
                    "answer": solution_str,         # 用于完全匹配
                    "original_puzzle": puzzle_str   # 用于验证解是否符合原题约束
                })
            },
            "extra_info": {
                "index": i,
                "split": "eval",
                "difficulty": data.get('metadata', {}).get('difficulty', 'standard')
            }
        }

        # 全部加入 eval
        eval_dataset.append(sample)

    # ================= 保存文件 =================
    
    # 转换为 HuggingFace Dataset 对象
    eval_ds = Dataset.from_list(eval_dataset)

    # 1. 保存为 Parquet
    parquet_path = f"{base_path}/{prefix}_eval_{filename_suffix}.parquet"
    eval_ds.to_parquet(parquet_path)
    
    # 2. 保存为 JSONL
    jsonl_path = f"{base_path}/{prefix}_eval_{filename_suffix}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in eval_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # ================= 验证输出 =================
    print(f"\n生成完成！")
    print(f"Eval Set Size: {len(eval_dataset)}")
    print(f"Parquet saved to: {parquet_path}")
    print(f"JSONL saved to:   {jsonl_path}")
    
    print("\n[Example Prompt]:")
    print("-" * 40)
    print(eval_ds[0]['prompt'][0]['content'])
    print("-" * 40)
    print("[Example Ground Truth]:")
    print(eval_ds[0]['reward_model']['ground_truth'])

if __name__ == "__main__":
    generate_sudoku_eval_set()