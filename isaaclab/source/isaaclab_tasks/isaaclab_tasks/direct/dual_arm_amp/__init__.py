import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
print("registering Isaac-Amp-Motion-Imitator-Direct-v0")
gym.register(
    id="Isaac-Amp-Motion-Imitator-Direct-v0",
    entry_point=f"{__name__}.amp_env:AmpMotionImitatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_env:AmpMotionImitatorEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)


