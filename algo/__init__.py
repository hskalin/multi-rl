from .ppo import PPO_agent
from .sac import SAC_Agent

# Mappings from strings to environments
algo_map = {"PPO": PPO_agent, "SAC": SAC_Agent}
