"""
This script generates music samples using the trained Birdie model.
It loads the checkpoint, samples feature sequences, and saves them.
"""

# Import required libraries
import os
import torch
import numpy as np
from birdie_rl import Birdie
import accelerate
import utils

# ⚡ CONFIGURATION ⚡
config = {
    "reward_fn": utils.reward_fn,
    "ds": utils.music_data_generator,  
    "objectives": utils.dummy_objectives if hasattr(utils, "dummy_objectives") else [{"name": "autoencoding"}],
    "tokenizer": None,
    "batch_size": 1,                        # 🔥 Generate one sample at a time
    "sequence_length": 512,
    "num_workers": 1,
    "steps_between_evaluations": 32,
    "num_steps": 10,                        # dummy (not training here)
    "accelerator": accelerate.Accelerator(),
    "music_mode": True,
    "data_path": "/Users/vishesh/Documents/University/Spring-25/Assignment-2/MusicGen-Birdie/birdie-main/music_dataset.npy",
}

# OUTPUT DIRECTORY
output_dir = "generated_music_samples"
os.makedirs(output_dir, exist_ok=True)

# LOAD BIRDIE
birdie = Birdie(config)

print(f"✅ Birdie model loaded!")

# GENERATE MUSIC SAMPLES
num_samples = 10  # 🔥 Number of music samples you want to generate

for i in range(num_samples):
    sample = birdie.get_next_training_sample()

    # Save input_ids (which are your generated MFCC feature sequences)
    generated_features = sample["input_ids"].detach().cpu().numpy()

    # Save as .npy file
    sample_save_path = os.path.join(output_dir, f"sample_{i}.npy")
    np.save(sample_save_path, generated_features)

    print(f"✅ Saved generated sample {i} at {sample_save_path}")

# ✅ Done
print(f"\n🎵 All {num_samples} music samples saved in {output_dir}/")

birdie.close()

# Hard exit to kill background threads
os._exit(0)
