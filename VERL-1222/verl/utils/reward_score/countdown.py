import json
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
import re
from . import utils


_num_re = re.compile(r"\b\d+\b")  # pre-compile once, reuse


def _extract_ints(expr_str: str) -> list[int]:
    """
    Fast path: grab the literal integers that appear in the source text.
    Handles duplicates correctly (e.g. “1 + 1 + 81” ⇒ [1, 1, 81]).
    """
    return [int(m) for m in _num_re.findall(expr_str)]


def compute_score(predict_str: str, ground_truth: str) -> float:
    # https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/games/countdown.py#L198

    gts = json.loads(ground_truth)
    reward = -1  # Default reward
    format_reward = 0
    try:
        answer = utils.extract_answer(predict_str)
    except:
        answer = None

    if answer is None or not answer.strip():
        return {"main": -1, "format": 0}

    try:
        user_answer = float(parse_expr(answer))
        used_numbers = _extract_ints(answer)
        target_numbers = gts["numbers"]

        if sorted(used_numbers) == sorted(target_numbers):
            format_reward = 1
            if np.isclose(user_answer, gts["answer"], atol=1e-6):
                reward = 1
        
    except Exception:
        reward = -1
    
    return {"main": reward, "format": format_reward}
    