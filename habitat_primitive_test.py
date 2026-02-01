#用于测试机器人最基础的动作（前进、转向、停止）是否在仿真器中生效
import habitat_sim
import cv2
import numpy as np

def main():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    # 使用內置的原始球體舞台
    sim_cfg.scene_id = "basis_sphere_uv_id_stage" 
    
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.resolution = [480, 640]
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    try:
        with habitat_sim.Simulator(cfg) as sim:
            print("✅ 成功加載原始幾何場景！正在開啟窗口...")
            for _ in range(100):
                obs = sim.get_sensor_observations()
                rgb = obs["color_sensor"]
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                cv2.imshow("RTX 4060 Habitat Test", bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"❌ 錯誤: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
