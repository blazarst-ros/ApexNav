from llm.client.deepseek_answer import deepseek_respond
from llm.utils.only_answer import only_answer
from llm.client.ollama_answer import ollama_respond
import os
def get_answer(client, prompt=None):
    """
    加固后的获取答案函数：
    确保返回的 similar_answer 永远是 list，respond 永远是 str。
    """
    # 1. 预设初始值，确保在任何异常情况下都有合法的返回值
    respond = ""
    similar_answer = []

    try:
        # 2. 逻辑分支对齐：使用 if-elif-else 确保逻辑互斥
        if client.llm_client == 'deepseek':
            # 调用你已经充值并加固过的 deepseek_respond
            respond = deepseek_respond(prompt=prompt)
            
        elif client.llm_client == 'ollama':
            respond = ollama_respond(model=client.ollama, prompt=prompt)
            
        else:
            print(f"Warning: Unknown llm_client type: {client.llm_client}")
            respond = ""

        # 3. 类型强制检查：防止底层接口意外返回 None
        if respond is None:
            respond = ""
        else:
            respond = str(respond)

        # 4. 提取动作：将字符串解析为列表
        # 即使 only_answer 内部出错返回 None，我们也在这里兜底
        parsed_result = only_answer(respond)
        
        if isinstance(parsed_result, list):
            similar_answer = parsed_result
        else:
            # 如果 only_answer 返回了 None 或其他非列表对象，强制转为空列表
            similar_answer = []

    except Exception as e:
        # 5. 全局异常捕获：确保即便代码逻辑出错，主程序也不会崩溃闪退
        print(f"Critical error in get_answer: {e}")
        similar_answer = []
        respond = ""

    # 最终检查：如果列表为空，可以视情况加入一个默认动作，防止主程序 subscriptable 报错
    # 如果主程序代码里有类似 similar_answer[0] 的操作，这里加一个 "stop" 最安全
    if not similar_answer:
        similar_answer = ["stop"]

    return similar_answer, respond
