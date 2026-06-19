"""Minimal test: load only the room shell USD with no rigid objects."""
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Minimal room shell test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# --- Imports after app launcher ---
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
def dummy_obs(env: ManagerBasedEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 1), device=env.device)
@configclass
class MinimalSceneCfg(InteractiveSceneCfg):
    """Only the room shell ? no rigid objects."""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )
    room_shell = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RoomShell",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/cezar/isaac-sim/isaac-projects/new_base_room.usda",
        ),
    )
@configclass
class DummyActionsCfg:
    pass
@configclass
class DummyObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        dummy = ObservationTermCfg(func=dummy_obs)
    policy: PolicyCfg = PolicyCfg()
@configclass
class MinimalEnvCfg(ManagerBasedEnvCfg):
    scene: MinimalSceneCfg = MinimalSceneCfg(num_envs=1, env_spacing=16.0)
    def __post_init__(self):
        self.decimation = 2
        self.sim.dt = 1.0 / 120.0
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (-7.0, -7.5, 0.78)
def main():
    env_cfg = MinimalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.actions = DummyActionsCfg()
    env_cfg.observations = DummyObservationsCfg()
    print("[INFO] Creating minimal env (room shell only)...", flush=True)
    env = ManagerBasedEnv(cfg=env_cfg)
    print("[INFO] SUCCESS ? Minimal env created!", flush=True)
    step = 0
    while simulation_app.is_running() and step < args_cli.max_steps:
        with torch.inference_mode():
            if step % 50 == 0:
                env.reset()
            env.step(action=torch.empty(env.num_envs, 0, device=env.device))
            step += 1
    print(f"[INFO] Completed {step} steps successfully.", flush=True)
    env.close()
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
