import habitat_sim
import cv2
import numpy as np
import os

def main():
    # 檢查文件是否存在
    scene_path = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    if not os.path.exists(scene_path):
        print(f"❌ 找不到地圖文件: {scene_path}")
        return

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene_path
    
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.resolution = [480, 640]
    sensor_spec.position = [0.0, 1.5, 0.0] # 站在地板上 1.5 米處
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    try:
        with habitat_sim.Simulator(cfg) as sim:
            print("✅ 地圖加載成功！正在開啟 3D 渲染視窗...")
            print("💡 提示：在視窗按下 'q' 鍵可關閉")
            
            # 渲染 100 幀
            for _ in range(500):
                obs = sim.get_sensor_observations()
                rgb = obs["color_sensor"]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                
                # 畫一個簡單的提示文字
                cv2.putText(bgr, "RTX 4060 Rendering Success!", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("ApexNav Habitat View", bgr)
                
                # 讓相機緩慢轉動
                agent = sim.get_agent(0)
                state = agent.get_state()
                state.rotation *= np.array([0, 0.01, 0, 1]) 
                agent.set_state(state)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"❌ 渲染運行錯誤: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
