import os
import sys
import random
import argparse
from glob import glob
import concurrent.futures

import torch
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from extract_embs_last_layer.f0_predictor import get_f0_predictor


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * \
        np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * \
        (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (
        f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min(
    ) >= 1, (f0_coarse.max(), f0_coarse.min())
    return f0_coarse


def extract_pitch(pitch_predictor, input_path, output_dir):
    try:
        output_path = os.path.join(output_dir, os.path.basename(input_path).replace(".wav", ".pt"))
        pitch = pitch_predictor.compute_f0(wavfile.read(input_path)[1])

        coarse_pitch = f0_to_coarse(pitch)

        if not isinstance(coarse_pitch, torch.Tensor):
            coarse_pitch = torch.tensor(coarse_pitch)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(coarse_pitch.cpu(), output_path)

        return output_path
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return None


if __name__ == "__main__":
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
        '--num-workers',
        type=int,
        default=1
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip existing pitch files"
    )
    args = parser.parse_args()

    hop_length = 320
    sampling_rate = 16000
    output_dir_name = "pitch_rmvpe"
    input_dir = os.path.join(args.base_dir, args.input_dir_name)
    output_dir = os.path.join(args.base_dir, output_dir_name)

    if args.device == "cuda" and args.num_workers > 1:
        print("Warning: Multiprocessing with CUDA is not supported. Setting num_workers to 1.")
        args.num_workers = 1

    pitch_predictor = get_f0_predictor(
        sampling_rate=sampling_rate,
        hop_length=hop_length,
        device=args.device,
        threshold=0.05
    )

    filelist = glob(os.path.join(input_dir, "**", "*.wav"), recursive=True)

    print(f"Found {len(filelist)} files")

    if args.num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(args.num_workers) as executor:
            futures = [
                executor.submit(
                    extract_pitch,
                    pitch_predictor,
                    file_path,
                    output_dir,
                ) for file_path in filelist
            ]
            for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting pitch"):
                result = future.result()
                # Optionally, handle the result (e.g., log successful processing)
                # For now, we ignore it as tqdm handles the progress
                pass
    else:
        for file_path in tqdm(filelist):
            extract_pitch(
                pitch_predictor,
                file_path,
                output_dir,
            )