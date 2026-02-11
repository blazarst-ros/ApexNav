from llm.utils.get_sys_prompt import get_similar_answer_prompt
from openai import OpenAI
client = OpenAI(api_key="sk-46f16f6dc6614a298138133a76a6b15c", base_url="https://api.deepseek.com")

"""
    底层：LLM 模型调用层
    输入：文本 Prompt：单个物体标签字符串
    返回：LLM 会返回自然语言文本（无固定格式，仅遵循 Prompt 约束），并非标准列表形式
    """
def deepseek_respond(prompt):
    system_prompts = get_similar_answer_prompt()
    msg = {
        "role": "user",
        "content": prompt
    }
    history = system_prompts + [msg]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        stream=False     # 非流式响应（一次性返回结果）
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    """当你直接运行 deepseek_answer.py 时（比如 python deepseek_answer.py），
    这个代码块才会执行，用于测试 API 是否能正常调用
    被其他文件调用时候，代码块完全不执行
    """
    print("正在测试 DeepSeek API...")
    result = deepseek_respond('dining table')
    print(f"API 回复内容: {result}")
