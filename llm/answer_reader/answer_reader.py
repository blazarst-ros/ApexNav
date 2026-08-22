from llm.answer import get_answer
from llm.answer_reader.structured_feedback import (
    append_feedback_record,
    build_feedback_record,
)
from llm.utils.answer_validation import (
    DEFAULT_FUSION_SCORE,
    DEFAULT_ROOM,
    get_fallback_answer,
    is_valid_legacy_answer,
    split_legacy_answer,
)
import ast
import os
"""业务解包层,
从结构化列表中解包出ApexNav 执行层可直接使用的 3 个业务参数，是连接 LLM 与导航核心逻辑的关键
"""


def _is_offline_client(llm_client):
    """Accept the config object used by Hydra as well as simple test values."""
    if isinstance(llm_client, str):
        return llm_client == "offline"
    return getattr(llm_client, "llm_client", None) == "offline"


def read_answer(llm_answer_path, llm_response_path, label, llm_client):
    label_existing = False
    llm_answer = None
    response = ""

    # 1. 读取缓存逻辑
    if os.path.exists(llm_answer_path):
        with open(llm_answer_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(f"{label}:"):
                    try:
                        # 缓存仅允许历史答案文件使用的 Python 字面量。
                        raw_content = line[len(label) + 1 :].strip()
                        cached_answer = ast.literal_eval(raw_content)
                        if is_valid_legacy_answer(cached_answer):
                            label_existing = True
                            llm_answer = cached_answer
                            print(f"Already have Answer for {label}: {llm_answer}")
                            break
                        print(f"Warning: Invalid cached Answer for {label}: {cached_answer}, will request LLM again")
                    except Exception as e:
                        print(f"Error parsing cache for {label}: {e}")
                        label_existing = False

    # 2. 获取新答案逻辑
    if not label_existing or llm_answer is None:
        if _is_offline_client(llm_client):
            print(f"No cached offline answer for {label}; using neutral defaults")
            return [], DEFAULT_ROOM, DEFAULT_FUSION_SCORE

        # 调用 get_answer，确保返回的是经过 only_answer 处理的列表
        llm_answer, response = get_answer(prompt=label, client=llm_client)
        if not is_valid_legacy_answer(llm_answer):
            print(f"Warning: LLM returned invalid Answer for {label}: {llm_answer}, using deterministic fallback")
            llm_answer = get_fallback_answer(label)
        
        # 写入缓存
        with open(llm_answer_path, "a+") as f:
            f.write(f"\n{label}: {llm_answer}")
        append_feedback_record(
            llm_answer_path,
            build_feedback_record(label, llm_answer, response),
        )
        
        with open(llm_response_path, "a+") as response_file:
            response_file.write(f"\n{label}: {response}")
        
        print(f"New Answer for {label}: {llm_answer}")

    # 3. 【核心修复】健壮的倒序解包逻辑
    # 目标结构: [action1, action2, ..., fusion_score(float), room(str)]
    
    llm_answer, room, fusion_score, score_found = split_legacy_answer(llm_answer)
    if room == DEFAULT_ROOM:
        print(f"Warning: Room name missing for {label}, using '{DEFAULT_ROOM}'")
    if not score_found:
        print(f"Warning: Score missing for {label}, using default {DEFAULT_FUSION_SCORE}")

    # 4. 确保最后剩下的 llm_answer 至少有一个动作
    if not llm_answer:
        llm_answer = get_fallback_answer(label)
        llm_answer, room, fusion_score, score_found = split_legacy_answer(llm_answer)

    return llm_answer, room, fusion_score
