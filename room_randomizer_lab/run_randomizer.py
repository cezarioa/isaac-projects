# run_randomizer.py
# Example script to launch the Isaac Lab simulation, instantiate the 
# room environment, and step through episodes to see the randomization.

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 1. Setup Isaac Lab app launcher (must happen before importing torch or isaaclab)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the Hospital Room Randomizer.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many simulation steps. Use 0 to run until closed.")
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


def dummy_observation(env: ManagerBasedEnv) -> torch.Tensor:
    """Single zero observation so ObservationManager has a valid policy group."""
    return torch.zeros((env.num_envs, 1), device=env.device)

@configclass
class DummyActionsCfg:
    """Empty action space."""
    pass

@configclass
class DummyObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        dummy = ObservationTermCfg(func=dummy_observation)
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
    print("[INFO] Initializing ManagerBasedEnv...", flush=True)
    env = ManagerBasedEnv(cfg=env_cfg)
    print("[INFO] ManagerBasedEnv initialized.", flush=True)

    # 5. Simulation Loop
    print("[INFO] Starting simulation loop. Press Ctrl+C to stop.", flush=True)
    step_count = 0
    reset_interval = 150  # Reset and randomize every 150 steps

    while simulation_app.is_running():
        with torch.inference_mode():
            # Reset environments periodically to see the randomizer in action
            if step_count % reset_interval == 0:
                print(f"[INFO] Triggering environment reset (step {step_count})...", flush=True)
                env.reset()
            
            # Step the simulation forward (with empty actions)
            env.step(action=torch.empty(env.num_envs, 0, device=env.device))
            
            step_count += 1
            if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
                print(f"[INFO] Reached --max_steps={args_cli.max_steps}; exiting.", flush=True)
                break

    env.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user.", flush=True)
    except SystemExit as exc:
        print(f"[ERROR] SystemExit during run: code={exc.code!r}", flush=True)
        raise
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
