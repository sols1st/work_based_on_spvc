import os
import h5py
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from Aebs.system.env import AebsEnv


# -----------------------------
# 1. Load data
# -----------------------------
fn = "./Aebs/data/Downsampled.h5"
with h5py.File(fn, 'r') as f:
    # Input state data
    y_data = np.array(f["y_train"], dtype=np.float32)

std1 = np.std(y_data)

# ------------------------------
# 2. PPO Training main loop
# ------------------------------
env = AebsEnv(std1)
eval_env = AebsEnv(std1)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    device="cpu",
)

script_dir = os.path.dirname(os.path.abspath(__file__)) 
model_save_path = os.path.join(script_dir, "ppo_aebs_controller.zip")  
checkpoint_dir = os.path.join(script_dir, 'logs')
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=checkpoint_dir,
    name_prefix='ppo_aebs'
)

eval_freq = 10000  # Evaluate every 50000 steps (can adjust according to timesteps)
n_eval_episodes = 3  # Run 3 episodes per evaluation and take the average

eval_callback = EvalCallback(
    eval_env=eval_env,                      
    best_model_save_path=os.path.join(script_dir, 'best_model'),  
    log_path=os.path.join(script_dir, 'logs/eval'),             
    eval_freq=eval_freq,                    
    n_eval_episodes=n_eval_episodes,        
    deterministic=True,                     
)

# -----------------------------
# 6. Start training
# -----------------------------
model.learn(
    total_timesteps=200000,
    callback=[
        checkpoint_callback,
        eval_callback, 
    ]
)

model.save(model_save_path)  
print(f"Model saved to script directory: {model_save_path}")
