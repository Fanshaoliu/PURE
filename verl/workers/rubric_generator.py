#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import openai
from openai import OpenAIError
import json
import requests

import re
import torch
from typing import List

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-rbulyxchzpldbaychbrwkrohkpesxnbfoamcwowsbyaezfvs"

# Configure API settings for the proxy service
# openai.api_key = "sk-LRfvpDFYRhJIZVhv2a2209084e854b98A3C1008bDe20Cb69"
openai.api_key = API_KEY
openai.api_base = "https://api.qqslyx.com/v1"

system_prompt="""
You are a master of solving mathematical problems. Based on the question and your own reasoning about a correct solution (without any external reference answer), you will classify a student's problem-solving steps into a series of Progress items. If a student problem-solving step contains an error, there will be an "error" label before that specific step in the Included Steps list. Progress items do not impose requirements on specific processes or very detailed sub-steps; instead, they serve as milestone-like checkpoints for solving the problem. Progress is progressive—only after accomplishing the previous Progress items can one move on to achieve the subsequent ones. You should infer a reasonable correct solution path yourself, and use it as a guide when designing the Progress items, rather than directly copying from the student's answer.

Inputs:

question: The full question text.

student_answer: A list of strings, where each string is one step or sentence from the student’s solution, in order.

Total items:

Return 1–6 progress items based on the complexity of the question.

Each progress item must include exactly four keys:

title (2–10 words)

description: A short sentence explaining this progress item. For example: This progress item means first clarifying that 1 minute equals 60 seconds.

weight: Assign values based on categories (Essential/Make Sense/Optional):

Essential → 3

Make Sense → 2

Optional → 1

Included Steps: A list of steps (strings) from the student's answer that align with this progress item.

Important rules for Included Steps:

Use the original student step text as elements in the Included Steps list.

If a step is mathematically incorrect, logically invalid, or based on a wrong assumption, then in Included Steps you must prefix it with "error: ".

Example: "error: I conclude that 7 is divisible by 2."

A single student step can belong to at most one Progress item. Do not duplicate the same step across multiple Progress items.

If a student step is completely irrelevant to solving the problem, you may omit it from all Progress items.

Category guidance:

Essential (weight = 3):
This progress item is critical and a mandatory conceptual step to solve the problem. Without it, the solution cannot logically reach the correct answer (even if the student accidentally guesses correctly).

Make Sense (weight = 2):
This progress item is highly effective and helpful for solving the problem and reflects a clear and meaningful intermediate step, but it is not strictly required in every valid solution path.

Optional (weight = 1):
This progress item is non-mandatory for solving the problem, but including it may improve the readability of the solution process, enhance the logical flow, or provide useful checks/interpretations.

Design principles for Progress items:

Think in terms of milestones along a correct solution path that you infer from the question itself.

Progress items should be ordered from early-stage understanding → intermediate reasoning → final conclusion.

Do not over-fragment: prefer a small number of meaningful milestones (1–6 in total).

If the problem requires a final explicit conclusion (e.g., “The final answer is (B)” or “The greatest common divisor is 320,000”), ensure there is an Essential progress item corresponding to reaching / stating the final answer.

If the question involves multiple-choice options, when relevant, explicitly state in the description things like “Identifies (A)” or “Chooses (C)” etc.

Format notes:

When referencing answer choices, explicitly state “Identifies (A)”, “Identifies (B)”, etc., in the description if relevant.

If reasoning must precede the final answer, include an Essential progress item for the reasoning before any progress item that corresponds to stating the final answer.

If conciseness is especially important in this problem, you may include an Optional progress item related to checking or simplifying the solution explanation.

Output:

Provide a JSON array of progress objects.
Each object must contain exactly four keys:
"title", "description", "weight", and "Included Steps".

"title": string

"description": string, and must start with the category prefix:

"Essential: ..." or

"Make Sense: ..." or

"Optional: ..."

"weight": integer 1, 2, or 3 (consistent with the category in the description).

"Included Steps": array of strings (student steps, some possibly prefixed by "error: " if they are incorrect).

Do not add any extra keys.

Now, based on the provided question and student_answer, infer a correct solution path in your mind and generate the progress items as outlined above.
"""

from typing import Any, List, Dict
def generate_progress_rubric(prompt_text: str, student_steps: List[str]) -> List[Dict[str, Any]]:
    """
    调用 DeepSeek，根据题目 + 学生解答步骤生成 Progress rubric。

    输入:
        prompt_text: 题目全文 (string)
        student_steps: 已经在外部按 _split_steps 切好的步骤列表 (List[str])

    输出:
        一个 JSON array 对应的 Python list，例如:
        [
          {
            "title": "...",
            "description": "Essential: ...",
            "weight": 3,
            "Included Steps": ["...", "error: ...", ...]
          },
          ...
        ]
        解析失败时返回 []。
    """
    if not isinstance(student_steps, list):
        raise ValueError("student_steps must be a list of strings")
    ordered_student_steps = [{i + 1: step} for i, step in enumerate(student_steps)]

    # 给 LLM 的 user 消息内容：按你 system_prompt 的约定是 question + student_answer
    user_payload = {
        "question": prompt_text,
        "student_answer": ordered_student_steps,
    }

    payload = {
        "model": "deepseek-ai/DeepSeek-V3.2-Exp",
        "messages": [
            {
                "role": "system",
                "content": system_prompt,  # 你上面定义好的 system_prompt
            },
            {
                "role": "user",
                # 用 JSON 防止 LLM 误解析结构
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
    }

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    })

    try:
        resp = session.post(API_URL, json=payload, timeout=200)
        resp.raise_for_status()
        raw_content = resp.json()["choices"][0]["message"]["content"]
        print("Rubric raw LLM output:\n", raw_content)

        # 只做最小必要的 JSON 提取，把输出变成真正的 JSON array
        def _extract_json_array(text: str):
            # 1) 直接尝试解析
            try:
                return json.loads(text)
            except Exception:
                pass

            # 2) 解析 ```json ... ``` 代码块
            m = re.search(r"```json(.*?)```", text, re.S)
            if m:
                candidate = m.group(1).strip()
                try:
                    return json.loads(candidate)
                except Exception:
                    pass

            # 3) 截取第一个 '[' 到最后一个 ']' 之间
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                candidate = text[start:end + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    pass

            raise ValueError("Cannot parse rubric JSON array from model output.")

        progress_items = _extract_json_array(raw_content)

        if not isinstance(progress_items, list):
            print("[Rubric] Parsed content is not a JSON array, fallback to [].")
            return []

        return progress_items

    except requests.exceptions.RequestException as e:
        print(f"[Rubric] HTTP error occurred: {e}")
    except Exception as e:
        print(f"[Rubric] Unexpected error: {e}")

    # 出错统一返回空列表，由外部决定 fallback（比如随机切分）
    return []