# run_randomizer.py
# Example script to launch the Isaac Lab simulation, instantiate the 
# room environment, and step through episodes to see the randomization.

import argparse

# 1. Setup Isaac Lab app launcher (must happen before importing torch or isaaclab)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the Hospital Room Randomizer.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports after app launcher
import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTermCfg, ObservationGroupCfg, ObservationTermCfg

from room_randomizer_lab.room_env_cfg import RoomEnvCfg

# --- Minimal Dummy Managers to satisfy ManagerBasedEnv ---
# Since you haven't defined what the RL agent controls or sees yet,
# we need dummy managers just so the environment initializes properly.

from isaaclab.utils import configclass

@configclass
class DummyActionsCfg:
    """Empty action space."""
    pass

@configclass
class DummyObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        pass
    policy: PolicyCfg = PolicyCfg()

# ---------------------------------------------------------

def main():
    # 3. Load the configuration
    env_cfg = RoomEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Inject dummy actions/observations so the environment can run
    env_cfg.actions = DummyActionsCfg()
    env_cfg.observations = DummyObservationsCfg()

    # 4. Initialize the environment
    print("[INFO] Initializing ManagerBasedEnv...")
    env = ManagerBasedEnv(cfg=env_cfg)

    # 5. Simulation Loop
    print("[INFO] Starting simulation loop. Press Ctrl+C to stop.")
    step_count = 0
    reset_interval = 150  # Reset and randomize every 150 steps

    while simulation_app.is_running():
        with torch.inference_mode():
            # Reset environments periodically to see the randomizer in action
            if step_count % reset_interval == 0:
                print(f"[INFO] Triggering environment reset (step {step_count})...")
                env.reset()
            
            # Step the simulation forward (with empty actions)
            env.step(action=torch.empty(env.num_envs, 0, device=env.device))
            
            step_count += 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user.")
    finally:
        simulation_app.close()
