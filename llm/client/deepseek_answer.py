def deepseek_respond(prompt):
    # 彻底删掉 try...except，让它报错就直接崩，这样我们就知道原貌了
    client = OpenAI(api_key='...', base_url='...')
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
