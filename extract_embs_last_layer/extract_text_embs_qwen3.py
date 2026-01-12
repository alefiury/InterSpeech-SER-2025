import os
import sys
import argparse
from os.path import exists, basename, join, relpath, dirname
from typing import Optional

import pandas as pd
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# try:
from nemo.collections.asr.models import ASRModel
# except ImportError:
#     print("NeMo is not installed. Please install NeMo to use ASRModel.")
#     ASRModel = None


def get_asr_model():
    if ASRModel is not None:
        return ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
    else:
        raise ImportError("NeMo is not installed. Please install NeMo to use ASRModel.")

@torch.inference_mode()
def get_transcript(asr_model, audio_filepath: str) -> str:
    if ASRModel is None:
        raise ImportError("NeMo is not installed. Please install NeMo to use ASRModel.")
    transcription = asr_model.transcribe([audio_filepath], source_lang='en', target_lang='en')
    return transcription[0]


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def load_model(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    model = AutoModel.from_pretrained(model_name, torch_dtype="auto")
    model = model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def extract_embedding_one(
    model,
    tokenizer,
    text: str,
    *,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    batch = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(**batch)

    emb = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])  # [1, D]
    emb = F.normalize(emb, p=2, dim=1)                                         # [1, D]
    return emb.squeeze(0).detach().cpu()                                       # [D]


def safe_stem(path_or_name: str) -> str:
    base = basename(str(path_or_name))
    stem, _ = os.path.splitext(base)
    return stem


def make_output_path(
    filename: str,
    input_dir: Optional[str],
    output_dir: str,
) -> str:
    if input_dir:
        try:
            rel = relpath(filename, input_dir)
            sub_dir = dirname(rel)
            out_subdir = join(output_dir, sub_dir)
        except Exception:
            out_subdir = output_dir
    else:
        out_subdir = output_dir

    os.makedirs(out_subdir, exist_ok=True)
    return join(out_subdir, safe_stem(filename) + ".pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="CSV with FileName and Transcript columns")
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--input-dir", default="", help="Optional root dir to preserve subfolders (like wav2vec script)")

    parser.add_argument("--file-col", default="FileName")
    parser.add_argument("--text-col", default="Transcript")
    parser.add_argument("--transcript-audio", action="store_true", help="If set, generate transcript from audio using ASR model")
    parser.add_argument("--base-dir", default=None, help="Audio files base directory, if needed for transcription")

    parser.add_argument("--use-instruct", action="store_true")
    parser.add_argument("--task", default="Represent the transcript for speech emotion recognition.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA is not available. Using CPU.")

    try:
        df = pd.read_csv(args.input_csv)
        print(df)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        sys.exit(1)

    if args.file_col not in df.columns:
        raise ValueError(f"CSV must contain columns: {args.file_col}")

    model, tokenizer = load_model(args.model_name, device)

    out_base = join(args.output_dir, "Texts_Qwen3_embeddings")
    os.makedirs(out_base, exist_ok=True)

    input_dir = args.input_dir.strip() or None

    if args.transcript_audio:
        asr_model = get_asr_model()

    saved = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting text embeddings"):
        fname = str(row[args.file_col])

        if args.transcript_audio:
            audio_path = os.path.join(args.base_dir, fname) if args.base_dir else fname
            text = get_transcript(asr_model, audio_path)
        else:
            text = str(row[args.text_col])

        if args.use_instruct:
            text = get_detailed_instruct(args.task, text)

        out_path = make_output_path(fname, input_dir, out_base)

        if exists(out_path):
            skipped += 1
            continue

        try:
            emb = extract_embedding_one(
                model, tokenizer, text,
                device=device,
                max_length=args.max_length
            )
            torch.save(emb, out_path)
            saved += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {fname}: {e}")

    print(f"Done. saved={saved}, skipped(existing)={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
