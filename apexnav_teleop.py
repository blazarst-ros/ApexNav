import habitat_sim
import cv2
import numpy as np
import os
import quaternion

def main():
    # 這是昨晚定位到的嵌套路徑，請確保路徑正確
    scene_path = "/home/blazarst/ApexNav/data/scene_datasets/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    
    if not os.path.exists(scene_path):
        print(f"❌ 找不到場景文件：{scene_path}")
        return

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = scene_path
    
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color"
    sensor_spec.resolution = [480, 640]
    sensor_spec.position = [0.0, 1.2, 0.0]
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    try:
        with habitat_sim.Simulator(cfg) as sim:
            agent = sim.get_agent(0)
            print("\n" + "="*40)
            print("🕹️  ApexNav 手動操控測試啟動！")
            print("W: 前進 | A: 左轉 | D: 右轉 | Q: 退出")
            print("="*40)
            
            while True:
                obs = sim.get_sensor_observations()
                rgb = obs["color"]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                
                cv2.putText(bgr, "Control: W/A/D | Exit: Q", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("ApexNav Explorer", bgr)
                
                key = cv2.waitKey(0) & 0xFF
                state = agent.get_state()
                
                if key == ord('w'):
                    forward_dir = state.rotation * quaternion.quaternion(0, 0, 0, -1) * state.rotation.conj()
                    state.position += np.array([forward_dir.x, 0, forward_dir.z]) * 0.2
                elif key == ord('a'):
                    state.rotation = state.rotation * quaternion.from_rotation_vector([0, 0.1, 0])
                elif key == ord('d'):
                    state.rotation = state.rotation * quaternion.from_rotation_vector([0, -0.1, 0])
                elif key == ord('q'):
                    break
                
                agent.set_state(state)
                
    except Exception as e:
        print(f"❌ 運行錯誤: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
