#测试 Habitat 的图形界面（GUI）是否能正常渲染
import habitat_sim
import cv2
import numpy as np

def make_cfg():
    # 1. 模擬器配置
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    # 使用內置測試場景，如果你有自己的 .glb 文件可以替換這裡
    sim_cfg.scene_id = "CHECKPOINT" 
    
    # 2. 感測器配置 (相機)
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [480, 640] # 設置解析度
    sensor_spec.position = [0.0, 1.5, 0.0] # 放在人的高度
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def main():
    cfg = make_cfg()
    try:
        with habitat_sim.Simulator(cfg) as sim:
            print("按 'q' 鍵退出窗口...")
            
            # 讓機器人隨機旋轉，我們看畫面的變化
            while True:
                # 取得隨機觀察值
                observations = sim.get_sensor_observations()
                rgb = observations["color_sensor"]
                
                # Habitat 輸出的是 RGBA，OpenCV 需要 BGR
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                
                # 彈出窗口顯示
                cv2.imshow("Habitat Robot View", bgr)
                
                # 旋轉機器人看風景
                agent = sim.get_agent(0)
                state = agent.get_state()
                state.rotation *= np.array([0, 0.1, 0, 1]) # 緩慢旋轉
                agent.set_state(state)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    except Exception as e:
        print(f"啟動失敗: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
