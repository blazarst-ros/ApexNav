import requests
import json

url = "https://api.deepseek.com/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-8b1d6c53577f4451a976842f1b09d9e0"
}
data = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 5
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
