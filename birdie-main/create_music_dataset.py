import os
import numpy as np
import librosa
from tqdm import tqdm

def extract_features(filepath, n_mfcc=20):
    y, sr = librosa.load(filepath, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Shape: [time_steps, n_mfcc]

def create_music_dataset(root_dir, save_path):
    dataset = []

    genres = os.listdir(root_dir)
    for genre in genres:
        genre_path = os.path.join(root_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"Processing genre: {genre}")
        for filename in tqdm(os.listdir(genre_path)):
            if filename.endswith('.wav'):
                file_path = os.path.join(genre_path, filename)
                try:
                    features = extract_features(file_path)
                    dataset.append({
                        "input": features,        # [time_steps, feature_dim]
                        "length": features.shape[0],
                        "genre": genre
                    })
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

    np.save(save_path, dataset)
    print(f"Dataset saved at {save_path}")

if __name__ == "__main__":
    data_folder = "/Users/vishesh/Documents/University/Spring-25/Assignment-1/Data"  # <- Change if your folder is different
    save_file = "music_dataset.npy"
    create_music_dataset(data_folder, save_file)
