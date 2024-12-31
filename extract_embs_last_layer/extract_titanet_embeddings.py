import os
import glob
import argparse
from typing import List, Tuple
from os.path import exists, basename, join, relpath, dirname

import wandb
import pandas as pd
from tqdm import tqdm
import torch, torchaudio
try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    nemo_asr = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    model = model.to(device)
    model.eval()
    return model

@torch.no_grad()
def extract_titanet_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str
) -> None:
    model = load_model()
    for filepath in tqdm(filelist, desc="Extracting embeddings"):
        # Load audio file
        # if not exists(filepath):
        #     print("file {} doesnt exist!".format(filepath))
        #     continue

        # Determine the relative path structure
        rel_path = relpath(filepath, input_dir)
        # Get the subdirectory structure
        sub_dir = dirname(rel_path)
        # Create the same subdirectory structure in output_dir
        output_subdir = join(output_dir, sub_dir)
        os.makedirs(output_subdir, exist_ok=True)

        emb = model.get_embedding(filepath)
        # print(emb.shape)
        output_filename = basename(filepath).split(".")[0] + ".pt"
        output_filepath = join(output_subdir, output_filename)
        torch.save(emb.cpu(), output_filepath)

wandb.login()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--base-dir",
        required=True,
        help="Path to the base directory"
    )
    parser.add_argument(
        "-i",
        "--input-dir-name",
        required=True,
        help="Name of the input directory, inside the base directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir-name",
        default="output_embeddings",
        help="Name of output directory",
    )
    parser.add_argument(
        "-c",
        "--input-csv",
        help="Metadata filepath",
    )
    parser.add_argument(
        "-col",
        "--column-name",
        default="filename",
        help="Column name of the csv file",
    )
    args = parser.parse_args()

    wandb.init(project="InterSpeech-SER-2025-Embeddings", entity="alefiury")

    input_dir = os.path.join(args.base_dir, args.input_dir_name)
    output_dir = os.path.join(args.base_dir, args.output_dir_name)

    filelist = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)

    assert len(filelist) > 0

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        filelist = df[args.column_name].tolist()

    os.makedirs(output_dir, exist_ok=True)

    extract_titanet_embeddings(filelist, input_dir, output_dir)


if __name__ == "__main__":
    main()