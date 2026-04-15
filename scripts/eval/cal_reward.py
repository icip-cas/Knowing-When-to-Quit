#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import json
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

sys.path.append("verl_path")
from verl.utils.reward_score import default_compute_score

tokenizer = AutoTokenizer.from_pretrained("Qwen3-8B")

NUM_WORKERS = 32

REFUSAL_KEYWORDS = [
    "Sorry, I can't solve this problem",
    "search_exhausted",
    "I cannot find a solution",
    "Here is how far I got",
    "impossible to reach the target",
    # "unknown" 
]

def is_unknown_response(text):
    if not isinstance(text, str): return False
    return any(keyword in text for keyword in REFUSAL_KEYWORDS)

def parallel_compute_scores_threadpool(sequences, gts, sources, prompts, abilities):
    n = len(sequences)
    results = [None] * n
    def _call(i):
        return default_compute_score(
            sources[i], sequences[i], gts[i],
            prompts[i] if prompts else None,
            abilities[i] if abilities else None,
            reward_config={},
        )
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = [ex.submit(_call, i) for i in range(n)]
        for i, fut in enumerate(futures): results[i] = fut.result()
    return results

def score_batch(batch):
    sequences, gts, sources, prompts, abilities = [], [], [], [], []
    for ins in batch:
        sequences.append(ins.get("pred", ""))
        gts.append(ins.get("reward_model", {}).get("ground_truth"))
        sources.append(ins.get("data_source", "unknown"))
        prompts.append(ins.get("prompt", ""))
        abilities.append(ins.get("ability", "math"))

    scores_objects = parallel_compute_scores_threadpool(sequences, gts, sources, prompts, abilities)

    for i, ins in enumerate(batch):
        score_obj = scores_objects[i]
        # 1. 真实正确性
        is_correct = bool(score_obj.get("main", 0) == 1)
        
        # 2. 拒答行为
        is_refused = is_unknown_response(sequences[i])
        if is_correct:
            is_refused = False
        
        # 3. 综合打分
        if is_correct:
            reliability_score = 1.0
        elif is_refused:
            reliability_score = 0.5 # 止损分
        else:
            reliability_score = 0.0 # 幻觉分

        ins["pass_list"] = [score_obj]
        ins["score"] = 1.0 if is_correct else 0.0
        ins["unk"] = is_refused
        ins["reliability_score"] = reliability_score
        
        ins["is_correct"] = is_correct
        ins["is_refused"] = is_refused
        # ins["len"] = len(tokenizer(sequences[i]).input_ids)
        ins["len"] = len(sequences[i])

    return batch

def main():
    for step in range(0,100,10):
        target_levels = list(range(4, 9, 2))


        base_path_tmpl = f"./{step}/unk_eval_{{}}number.jsonl"

        # 表头设计
        headers = [
            "Level", "Count", 
            "Acc",         # 正确率: Correct / Total
            "Rel Score",   # 综合分: (Correct + 0.5*ValidRefusal) / Total
            "Ref Rate",    # 拒答率: Refused / Total (新增)
            "Halluc Rate",  # 幻觉率: (Wrong & Answered) / Total_Wrong
            "Len"
        ]
        
        print(f"\n{'='*100}")
        # 调整了宽度以适应新列
        print(f"{headers[0]:<8} | {headers[1]:<6} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<10} | {headers[5]:<10} | {headers[6]:<10}")
        print(f"{'-'*100}")

        agg_stats = {"total": 0, "correct": 0, "rel_score_sum": 0, 
                    "wrong_total": 0, "valid_refusal": 0, "hallucination": 0, 
                    "total_refusal": 0, "bad_refusal": 0, "len": 0}

        for i in target_levels:
            input_path = base_path_tmpl.format(i)

            output_path = input_path.removesuffix(".jsonl") + ".score.jsonl"

            if not os.path.exists(input_path): continue
            
            all_items = []
            with open(input_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip(): all_items.append(json.loads(line))
            
            if not all_items: continue

            scored_batch = score_batch(all_items)

            # --- 统计 ---
            n_total = len(scored_batch)
            n_correct = sum(1 for x in scored_batch if x["is_correct"])
            n_wrong = n_total - n_correct
            
            # 拒答统计
            n_refused = sum(1 for x in scored_batch if x["is_refused"])
            n_valid_refusal = sum(1 for x in scored_batch if (not x["is_correct"]) and x["is_refused"])
            n_hallucination = sum(1 for x in scored_batch if (not x["is_correct"]) and (not x["is_refused"]))

            # 指标计算
            acc = n_correct / n_total
            rel_score = sum(x["reliability_score"] for x in scored_batch) / n_total
            avg_len = sum(x["len"] for x in scored_batch) / n_total
            
            # 新增: 拒答率 (Refusal Rate)
            ref_rate = n_refused / n_total
            
            # Hallucination Rate
            halluc_rate = (n_hallucination / n_wrong) if n_wrong > 0 else 0.0

            # 打印
            print(f"Level {i:<2} | {n_total:<6} | {acc:.4f}     | {rel_score:.4f}     | {ref_rate:.4f}     | {halluc_rate:.4f} | {avg_len:.4f}")

            # 写入文件
            with open(output_path, "w", encoding="utf-8") as fout:
                for item in scored_batch: fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            # 累加全局统计
            agg_stats["total"] += n_total
            agg_stats["correct"] += n_correct
            agg_stats["rel_score_sum"] += sum(x["reliability_score"] for x in scored_batch)
            agg_stats["wrong_total"] += n_wrong
            agg_stats["valid_refusal"] += n_valid_refusal
            agg_stats["hallucination"] += n_hallucination
            agg_stats["total_refusal"] += n_refused
            agg_stats["len"] += sum(x["len"] for x in scored_batch) 

        print(f"{'-'*100}")
        
        # 计算全局平均
        if agg_stats["total"] > 0:
            g_acc = agg_stats["correct"] / agg_stats["total"]
            g_rel = agg_stats["rel_score_sum"] / agg_stats["total"]
            
            # 全局拒答率
            g_ref_rate = agg_stats["total_refusal"] / agg_stats["total"]
            
            g_halluc = agg_stats["hallucination"] / agg_stats["wrong_total"] if agg_stats["wrong_total"] > 0 else 0

            g_len = agg_stats["len"] / agg_stats["total"]
            
            print(f"{'Overall':<8} | {agg_stats['total']:<6} | {g_acc:.4f}     | {g_rel:.4f}     | {g_ref_rate:.4f}     | {g_halluc:.4f} | {g_len:.4f}")

        print(f"{'='*100}\n")
        break


if __name__ == "__main__":
    main()
