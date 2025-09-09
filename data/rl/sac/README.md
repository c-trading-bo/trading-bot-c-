# SAC (Soft Actor-Critic) Models Directory
# This directory contains trained SAC policy models
# The OnnxModelLoader will monitor this directory for changes every 60 seconds

# Expected file formats:
# - *.zip (Stable-Baselines3 model archives)
# - *.pkl (Pickled models)  
# - *.pt (PyTorch model files)

# Example files:
# sac_es_policy_latest.zip
# sac_nq_policy_latest.zip
# critic_network_latest.pt
# actor_network_latest.pt

# When changes are detected, the system will:
# 1. Log the model update
# 2. Notify the Python SAC agent to reload
# 3. Update position sizing recommendations