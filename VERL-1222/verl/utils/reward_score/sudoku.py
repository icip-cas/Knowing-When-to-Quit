import json

def compute_score(solution_str, ground_truth):
    oracle_answer = json.loads(ground_truth)["answer"]
    answer = solution_str.split("</think>")[-1].strip()
    if "boxed" in answer:
        answer = answer.split("\\boxed{")[-1].split("}")[0]
    
    answer_stripped = "\n".join(l.rstrip() for l in answer.split("\n"))
    oracle_answer_stripped = "\n".join(l.rstrip() for l in oracle_answer.split("\n"))
    board_size = len(oracle_answer_stripped)

    if answer_stripped == oracle_answer_stripped:
        return {"main": 1, "format": 1}
    else:
        return {"main": -1, "format": 1}
    

    # else:
    #     # 2. accept answers with correct numeric sequence (ignoring non-numeric characters)
    #     row = 0
    #     num_matching = 0
    #     for ln in answer.split("\n"):
    #         if row >= board_size:
    #             break
    #         numbers = [int(c) for c in ln if c in "123456789"]
    #         if len(numbers) != board_size:
    #             continue  # ignore lines without numbers
    #         for a, b in zip(solution[row], numbers):
    #             if a == b:
    #                 num_matching += 1
    #         row += 1

    #     reward = num_matching / (board_size * board_size)
    #     reward *= 0.9  # penalty for not using standard format

    # if len(answer) > len(oracle_answer):
    #     reward *= len(oracle_answer) / len(answer)  # penalty for additional length
    # return reward
