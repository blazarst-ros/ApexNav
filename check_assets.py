#用于检查你下载的 HM3D 或 MP3D 数据集路径是否正确，资源文件是否完整。如果你换了数据集跑不动，可以运行它进行校验#

import os

def check_apexnav_assets():
    # 根據 README 定義的路徑結構
    base_dir = os.getcwd()
    assets = {
        "模型權重 (Model Weights)": {
            "data/mobile_sam.pt": "https://github.com/ChaoningZhang/MobileSAM",
            "data/groundingdino_swint_ogc.pth": "GroundingDINO 核心權重",
            "data/yolov7-e6e.pt": "YOLOv7 權重"
        },
        "場景數據集 (Scene Datasets)": {
            "data/scene_datasets/hm3d": "HM3D 數據 (需要申請權限)",
            "data/scene_datasets/mp3d": "MP3D 數據 (需要申請權限)"
        },
        "任務數據集 (Task Datasets)": {
            "data/datasets/objectnav/hm3d/v2": "HM3D 導航任務定義",
            "data/datasets/objectnav/mp3d/v1": "MP3D 導航任務定義"
        }
    }

    print("="*60)
    print("🔍 ApexNav 資源完整度檢查")
    print("="*60)

    all_passed = True
    for category, items in assets.items():
        print(f"\n【{category}】")
        for path, desc in items.items():
            full_path = os.path.join(base_dir, path)
            status = "✅ 存在" if os.path.exists(full_path) else "❌ 缺失"
            if "❌" in status:
                all_passed = False
            print(f"  {status} | {path.split('/')[-1]:<30} | {desc}")

    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有核心資源已就緒！你可以開始運行算法了。")
    else:
        print("💡 提示：請根據 README 中的指令下載缺失的資源。")
    print("="*60)

if __name__ == "__main__":
    check_apexnav_assets()
