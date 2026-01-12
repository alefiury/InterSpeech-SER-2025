import os
import pandas as pd
from tqdm import tqdm


def main():
    base_dir = "/hadatasets/alef.ferreira/SER/Interspeech/embeddings_qwen3/Texts_Qwen3_embeddings"
    # metadatapath = "/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/Dataset/train_set.csv"
    metadatapath = "/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/Dataset/validation_set.csv"

    df = pd.read_csv(metadatapath)
    missing_counter = 0

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Checking embeddings"):
        fname = str(row["FileName"])
        emb_path = os.path.join(base_dir, fname.replace(".wav", ".pt"))
        if not os.path.exists(emb_path):
            print(f"Missing embedding for file: {fname} at path: {emb_path}")
            missing_counter += 1


    print(f"Total missing embeddings: {missing_counter}")



if __name__ == "__main__":
    main()