import requests
import base64
import json
import cv2

def test_server(name, port, payload):
    url = f"http://localhost:{port}/detect"  # ApexNav 默认通常是 /detect 或 /process
    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code == 200:
            print(f"✅ [SUCCESS] {name} (Port {port}): {response.json()}")
        else:
            print(f"❌ [FAILED] {name} (Port {port}): Status {response.status_code}")
    except Exception as e:
        print(f"⚠️  [ERROR] {name} (Port {port}): 无法连接或超时 ({e})")

# 将图片转为 Base64 编码（这是 VLM 服务器通用的传输方式）
with open("test.jpg", "rb") as f:
    img_str = base64.b64encode(f.read()).decode('utf-8')

# --- 2. 开始测试 ---

# 测试 GroundingDINO (开放词汇检测)
test_server("GroundingDINO", 12181, {
    "image": img_str,
    "text_prompt": "chair . table . flower pot ." # 注意：DINO 通常用点号分隔标签
})

# 测试 YOLOv7 (常规物体检测)
test_server("YOLOv7", 12182, {
    "image": img_str,
    "conf_threshold": 0.25
})

# 测试 MobileSAM (分割)
# 注意：SAM 通常需要 Bounding Box 作为 Prompt
test_server("MobileSAM", 12183, {
    "image": img_str,
    "boxes": [[100, 100, 200, 200]] 
})

# 测试 BLIP-2 (图像-文本确认)
test_server("BLIP2", 12185, {
    "image": img_str,
    "text": "Is there a chair in this image?",
    "task": "vqa"
})
