import os
from pathlib import Path

from llm.utils.get_sys_prompt import get_similar_answer_prompt


def _get_api_key(env_file=None):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        if env_file is None:
            env_file = Path(__file__).resolve().parents[2] / ".env.local"
        if Path(env_file).exists():
            for line in Path(env_file).read_text().splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return api_key


def _get_client():
    from openai import OpenAI

    return OpenAI(api_key=_get_api_key(), base_url="https://api.deepseek.com")

def deepseek_respond(prompt):
    system_prompts = get_similar_answer_prompt()
    msg = {
        "role": "user",
        "content": prompt
    }
    history = system_prompts + [msg]
    client = _get_client()

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=history,
        stream=False
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    deepseek_respond('dining table')
