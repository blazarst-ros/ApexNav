import ast
import re

def only_answer(response):
    # 1. 基础防御
    if response is None or not isinstance(response, str):
        return [] 

    # 2. 正则匹配：提取 Answer: [...] 内部的内容
    match = re.search(r'Answer:\s*\[(.*?)\]', response, re.DOTALL)
    if not match:
        return []

    content = match.group(1)
    
    # 3. 智能解析逻辑
    # 策略：不手动拼接字符串，而是利用正则切分后再进行类型识别
    raw_items = [i.strip() for i in content.split(',')]
    processed_elements = []

    for item in raw_items:
        # a. 尝试识别数字 (int 或 float)
        if re.match(r'^-?\d+(\.\d+)?$', item):
            try:
                # 核心改进：直接存入数字对象，而不是字符串
                val = float(item) if '.' in item else int(item)
                processed_elements.append(val)
                continue
            except ValueError:
                pass
        
        # b. 识别布尔值 (如果有需要)
        if item.lower() in ['true', 'false']:
            processed_elements.append(item.lower() == 'true')
            continue

        # c. 处理字符串：去除多余引号并统一加上 Python 的单引号
        clean_item = item.strip("'").strip('"')
        processed_elements.append(clean_item)

    # 4. 逻辑完整性校验
    # 此时 processed_elements 已经是一个包含 [str, str, float, str] 的真实 Python 列表
    return processed_elements
