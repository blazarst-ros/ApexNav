from llm.client.deepseek_answer import deepseek_respond
from llm.utils.only_answer import only_answer
from llm.client.ollama_answer import ollama_respond
import os


def get_answer(client, prompt=None):
    """
    统一处理不同 LLM 客户端（DeepSeek/Ollama）的调用逻辑，
    并对模型返回结果做初步解析，
    向下调用 deepseek_answer.py/ollama_answer.py，
    向上为 answer_reader.py 提供标准化结果。
    是LLM/VLM 结果调度与解析文件
    """
    # 1. 初始值
    respond = ""
    similar_answer = []

    # --- 调试代码：强制打印以确认函数被调用 ---
    print(f"\n[DEBUG] get_answer 启动 | 目标模型: {getattr(client, 'llm_client', '未知')}")
    print(f"[DEBUG] 当前 Prompt 长度: {len(prompt) if prompt else 0}")

    try:
        # 2. 逻辑分支对齐
        if client.llm_client == 'deepseek':
            print("🚀 [LLM] 正在向 DeepSeek 官网发起实时请求...")
            respond = deepseek_respond(prompt=prompt)
            
        elif client.llm_client == 'ollama':
            print("🏠 [LLM] 正在调用本地 Ollama...")
            respond = ollama_respond(model=client.ollama, prompt=prompt)
            
        else:
            print(f"⚠️ Warning: 未知的 llm_client 类型: {client.llm_client}")
            respond = ""

        # 3. 类型强制检查
        respond = str(respond) if respond is not None else ""

        # 4. 提取动作
        parsed_result = only_answer(respond)  # 核心解析：从自然语言提取结构化列表(N(string)+1(float)+1(string))
        
        if isinstance(parsed_result, list):
            similar_answer = parsed_result
        else:
            similar_answer = []

    except Exception as e:
        print(f"❌ Critical error in get_answer: {e}")
        similar_answer = []
        respond = ""

    # 5. 最终安全性检查
    if not similar_answer:
        similar_answer = ["stop"]
    
    # 打印结果反馈
    print(f"✅ [LLM 回复成功] 动作解析为: {similar_answer}")

    return similar_answer, respond
