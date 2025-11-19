import os
import glob
import argparse
from typing import List, Tuple
from os.path import exists, basename, join, relpath, dirname

import pandas as pd
from tqdm import tqdm
import torch, torchaudio
try:
    from espnet2.tasks.ssl import SSLTask
except ImportError:
    SSLTask = None


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    model, _ = SSLTask.build_model_from_file(
        None,
        model_path,
    )
    model = model.to(device)
    model.eval()
    return model

@torch.inference_mode()
def extract_xeus_embeddings(
    filelist: List[str],
    input_dir: str,
    output_dir: str,
    model_path: str = "xeus/wav2vec2-xls-r-300m-lv60-self",
) -> None:
    model = load_model(model_path)
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

        audio_data, sr = torchaudio.load(filepath)
        # If stereo, convert to mono
        if audio_data.dim() > 1:
            audio_data = audio_data.mean(dim=0)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_data = resampler(audio_data)

        audio_data = audio_data.unsqueeze(0).to(device)
        wav_lengths = torch.tensor([audio_data.shape[-1]]).to(device)
        with torch.no_grad():
            hidden_states = model.encode(audio_data, wav_lengths, use_final_output=False)["encoder_output"][-1]
        output_filename = basename(filepath).split(".")[0] + ".pt"
        output_filepath = join(output_subdir, output_filename)
        torch.save(hidden_states.cpu(), output_filepath)


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
    parser.add_argument(
        "-m",
        "--model-path",
        default="ssl_checkpoints/XEUS/xeus_checkpoint_new.pth",
        help="Path or name of the pre-trained model",
    )
    args = parser.parse_args()

    input_dir = os.path.join(args.base_dir, args.input_dir_name)
    output_dir = os.path.join(args.base_dir, args.output_dir_name)

    filelist = glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)

    assert len(filelist) > 0

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        filelist = df[args.column_name].tolist()

    os.makedirs(output_dir, exist_ok=True)

    extract_xeus_embeddings(filelist, input_dir, output_dir, model_path=args.model_path)


if __name__ == "__main__":
    main()