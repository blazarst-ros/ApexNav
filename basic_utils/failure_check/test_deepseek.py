import sys
from openai import OpenAI
import requests

def verify_deepseek_connection():
    # 你的配置信息
    api_key = "sk-dedbf651b15c4f2792fe435398352009"
    base_url = "https://api.deepseek.com"
    
    print("--- 开始 DeepSeek 接口连通性测试 ---")
    
    # 1. 基础网络检查 (Network Connectivity)
    try:
        print(f"[步骤 1] 正在测试网络连接: {base_url}...")
        response = requests.get(base_url, timeout=5)
        print(f"✅ 网络接入成功 (HTTP 状态码: {response.status_code})")
    except Exception as e:
        print(f"❌ 网络连接失败: 请检查你的网络是否能访问该地址。错误: {e}")
        return

    # 2. 初始化 OpenAI 客户端
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 3. 发送 API 调用测试 (API Logic Verification)
    try:
        print(f"[步骤 2] 正在发送测试请求 (Model: deepseek-chat)...")
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Confirm if you are DeepSeek API."},
            ],
            stream=False
        )
        
        # 4. 验证返回内容
        answer = completion.choices[0].message.content
        print(f"✅ API 调用成功！")
        print(f"--- 机器人回复内容 --- \n{answer}\n----------------------")
        
        # 5. 检查 Token 使用情况 (用于确认余额/额度有效)
        print(f"[步骤 3] Token 消耗情况: {completion.usage}")
        
    except Exception as e:
        print(f"❌ API 调用失败！具体原因:")
        # 针对常见错误的分类逻辑
        error_msg = str(e).lower()
        if "auth" in error_msg or "401" in error_msg:
            print("   👉 错误原因：API Key 无效或已过期。")
        elif "insufficient_balance" in error_msg or "402" in error_msg:
            print("   👉 错误原因：账户余额不足或额度用尽。")
        elif "timeout" in error_msg:
            print("   👉 错误原因：请求超时，请尝试调大 client 的 timeout 参数。")
        else:
            print(f"   👉 详细报错信息: {e}")

if __name__ == "__main__":
    verify_deepseek_connection()
