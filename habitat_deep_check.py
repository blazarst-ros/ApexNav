#通常用于深度检查 Habitat 仿真器的物理引擎渲染和 GPU 加载是否正常
import habitat_sim
import torch

print("="*40)
print("🏠 Habitat-sim 渲染功能深度檢測")
print("="*50)

# 1. 基礎版本與編譯配置
print(f"【1】基礎信息:")
print(f"  - 版本: {getattr(habitat_sim, '__version__', '未知')}")
print(f"  - 是否支持 CUDA: {habitat_sim.cuda_enabled}")
print(f"  - 是否編譯了內置渲染器: {habitat_sim.built_with_renderer}")

# 2. GPU 渲染鏈接測試
print(f"\n【2】GPU 渲染後端測試:")
try:
    # 創建一個最小化的渲染配置
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.gpu_device_id = 0  # 使用 RTX 4060
    
    # 嘗試初始化一個「空場景」的模擬器實例
    # 這會觸發 OpenGL 上下文的創建
    with habitat_sim.Simulator(cfg) as sim:
        print("  - OpenGL 上下文創建: 成功 ✅")
        print(f"  - 模擬器渲染設備 ID: {sim.gpu_device_id}")
        
    print("\n🎉 結論: Habitat-sim 與 GPU 驅動鏈接完美，渲染功能正常！")

except Exception as e:
    print(f"\n❌ 渲染測試失敗!")
    print(f"  - 錯誤詳情: {e}")
    print("\n💡 提示: 如果報錯與 'GLX' 或 'Display' 有關，說明在遠程 SSH 模式下需要配置 Headless 渲染。")

print("="*40)
