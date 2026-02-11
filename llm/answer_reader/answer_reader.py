from llm.answer import get_answer
import os
"""业务解包层,
从结构化列表中解包出ApexNav 执行层可直接使用的 3 个业务参数，是连接 LLM 与导航核心逻辑的关键
"""
def read_answer(llm_answer_path, llm_response_path, label, llm_client):
    label_existing = False
    llm_answer = None

    # 1. 读取缓存逻辑
    if os.path.exists(llm_answer_path):
        with open(llm_answer_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(f"{label}:"):
                    label_existing = True
                    try:
                        # 使用 eval 解析存入的列表字符串
                        raw_content = line[len(label) + 1 :].strip()
                        llm_answer = eval(raw_content)
                        print(f"Already have Answer for {label}: {llm_answer}")
                    except Exception as e:
                        print(f"Error parsing cache for {label}: {e}")
                        label_existing = False
                    break

    # 2. 获取新答案逻辑
    if not label_existing or llm_answer is None:
        # 调用 get_answer，确保返回的是经过 only_answer 处理的列表
        llm_answer, response = get_answer(prompt=label, client=llm_client)
        
        # 写入缓存
        with open(llm_answer_path, "a+") as f:
            f.write(f"\n{label}: {llm_answer}")
        
        with open(llm_response_path, "a+") as response_file:
            response_file.write(f"\n{label}: {response}")
        
        print(f"New Answer for {label}: {llm_answer}")

    # 3. 【核心修复】健壮的倒序解包逻辑
    # 目标结构: [action1, action2, ..., fusion_score(float), room(str)]
    
    room = "unknown"
    fusion_score = 0.4 # 默认阈值

    # 解包 Room (期待最后一个是 str)
    if len(llm_answer) > 0:
        last_item = llm_answer[-1]
        if isinstance(last_item, str) and last_item not in ['move_forward', 'turn_left', 'turn_right', 'stop']:
            room = llm_answer.pop()
        else:
            # 如果最后一个元素是动作而不是房间名，这里做个兼容
            print(f"Warning: Room name missing for {label}, using 'unknown'")
    
    # 解包 Score (期待现在最后一个是 float/int)
    if len(llm_answer) > 0:
        last_item = llm_answer[-1]
        # 尝试将最后一个元素转为 float
        try:
            if isinstance(last_item, (float, int)):
                fusion_score = float(llm_answer.pop())
            elif isinstance(last_item, str) and last_item.replace('.', '', 1).isdigit():
                fusion_score = float(llm_answer.pop())
            else:
                print(f"Warning: Score missing for {label}, using default 0.5")
        except:
            print(f"Warning: Error parsing score, using default 0.5")

    # 4. 确保最后剩下的 llm_answer 至少有一个动作
    if not llm_answer:
        llm_answer = ["stop"]

    return llm_answer, room, fusion_score
