import reasoning_gym
from datasets import Dataset
import json
from tqdm import tqdm

# 原始 template
# TEMPLATE = """Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Return the final answer in \\boxed{{}}, for example \\boxed{{(1 + 2) / 3}}. 

# Inside \\boxed{{}}, you may only use:
#    * digits `0–9`
#    * the operators `+`, `-`, `*`, `/`
#    * parentheses `(` and `)`
#    * optional spaces

# Do NOT use any other LaTeX commands or symbols, such as `\\times`, `\\frac`, etc.
# """

# # 改进后加了一些格式约束
# TEMPLATE = """Using the numbers {numbers}, write a **single arithmetic expression** whose value is exactly {target}.

# Rules:
# 1. You may only use the four basic operations: +, -, *, /
# 2. Each given number must be used **once**.
# 3. Your answer must be **only** the final expression, wrapped in \\boxed{{...}}  
#    For example: \\boxed{{(1 + 2) / 3}}

# Inside \\boxed{{...}}, you may use ONLY:
#    * digits 0–9
#    * the operators +, -, *, /
#    * parentheses ( and )
#    * optional spaces

# Do NOT use:
#    * any other LaTeX commands or symbols (such as \\times, \\frac, \\cdot, ^, etc.)
#    * any text or explanation outside of the single \\boxed{{...}} expression.
# """

# 显式unknown 版
TEMPLATE = """Using the numbers {numbers}, write a **single arithmetic expression** whose value is exactly {target}.

Rules:
1. You may only use the four basic operations: +, -, *, /
2. Each given number must be used **once**.
3. Your answer must be **only** the final expression, wrapped in \\boxed{{...}}  
   For example: \\boxed{{(1 + 2) / 3}}

Inside \\boxed{{...}}, you may use ONLY:
   * digits 0–9
   * the operators +, -, *, /
   * parentheses ( and )
   * optional spaces

Do NOT use:
   * any other LaTeX commands or symbols (such as \\times, \\frac, \\cdot, ^, etc.)
   * any text or explanation outside of the single \\boxed{{...}} expression.

Important:
If, after several reasonable attempts, you believe you cannot find a correct expression, do NOT keep searching or loop in your reasoning. Instead, do NOT output \\boxed{{...}}. Reply in plain text starting with:
"Sorry, I can't solve this problem. Here is how far I got: ..."
and then briefly summarize how far your reasoning has progressed so far.
"""




for number in [3, 4, 5, 6, 7, 8]:
    dataset = reasoning_gym.create_dataset(
        'countdown',
        min_numbers=number,
        max_numbers=number,
        min_value=1,
        max_value=100,
        min_target=1,
        max_target=999,
        operators=("+", "-", "*", "/"),
        shuffle=True,
        seed=42,
        size=1100
    )

    train_dataset = []
    eval_dataset = []

    for i, data in enumerate(tqdm(dataset)):
        sample = {
            "data_source": f"reasoning-gym-countdown-depth{number}",
            "prompt": [{
                "role": "user",
                "content": TEMPLATE.format(numbers=str(data['metadata']['numbers']), target=data['metadata']['target'])
            }],
            "ability": "reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({
                    "answer": data['metadata']['target'],
                    "numbers": data['metadata']['numbers']
                })
            },
            "extra_info": {
                "index": i,
                "split": "train" if i < 1000 else "eval"
            }
        }

        if i < 1000:
            train_dataset.append(sample)
        else:
            eval_dataset.append(sample)

    # Save to parquet
    train_ds = Dataset.from_list(train_dataset)
    eval_ds = Dataset.from_list(eval_dataset)

    print("Example prompt:", train_ds[0]['prompt'][0]['content'])

    base_path = "./reasoning-unk/data"
    
    prefix = "unk"
    # Save as parquet
    train_ds.to_parquet(f"{base_path}/{prefix}_train_{number}number.parquet")
    eval_ds.to_parquet(f"{base_path}/{prefix}_eval_{number}number.parquet")
    
    # Save as jsonl
    with open(f"{base_path}/{prefix}_train_{number}number.jsonl", 'w', encoding='utf-8') as f:
        for item in train_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(f"{base_path}/{prefix}_eval_{number}number.jsonl", 'w', encoding='utf-8') as f:
        for item in eval_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {number}-number datasets: {len(train_dataset)} train, {len(eval_dataset)} eval samples")