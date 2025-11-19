#!/bin/bash
CUDA_DEVICE=0
BASE_DIR="/mnt/EA3C54E23C54AB79/Datasets/augmented_audios"
INPUT_DIR="reference"

# Wav2vec2
MODEL_NAME="wav2vec2-large"
echo "Extracting embeddings for model: $MODEL_NAME"
OUTPUT_DIR="wav2vec2-large"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_wav2vec_embeddings.py \
    -b=$BASE_DIR \
    -i=$INPUT_DIR \
    -o=$OUTPUT_DIR \
    -m=$MODEL_NAME

# Whisper
MODEL_NAME="whisper-large-v3"
echo "Extracting embeddings for model: $MODEL_NAME"
OUTPUT_DIR="whisper-large-v3"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_whisper_embeddings.py \
    -b=$BASE_DIR \
    -i=$INPUT_DIR \
    -o=$OUTPUT_DIR \
    -m=$MODEL_NAME

# Wav2BERT 2.0
echo "Extracting embeddings for model: Wav2BERT 2.0"
OUTPUT_DIR="wav2bert-large"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_wav2bert_embeddings.py \
    -b=$BASE_DIR \
    -i=$INPUT_DIR \
    -o=$OUTPUT_DIR

# WavLM
MODEL_NAME="wavlm-large"
echo "Extracting embeddings for model: $MODEL_NAME"
OUTPUT_DIR="wavlm-large"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_wavlm_embeddings.py \
    -b=$BASE_DIR \
    -i=$INPUT_DIR \
    -o=$OUTPUT_DIR \
    -m=$MODEL_NAME

# HUBERT
MODEL_NAME="hubert-large-ll60k"
echo "Extracting embeddings for model: $MODEL_NAME"
OUTPUT_DIR="hubert-large"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_hubert_embeddings.py \
    -b=$BASE_DIR \
    -i=$INPUT_DIR \
    -o=$OUTPUT_DIR \
    -m=$MODEL_NAME

# XEUS
echo "Extracting embeddings for model: XEUS"
OUTPUT_DIR="xeus"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_xeus_embeddings.py \
    -b=$BASE_DIR \
    -i=$INPUT_DIR \
    -o=$OUTPUT_DIR

# NEST
echo "Extracting embeddings for model: NEST"
OUTPUT_DIR="nest"
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python3 extract_nest_embeddings.py \
    -b=$BASE_DIR \
    -i=$INPUT_DIR \
    -o=$OUTPUT_DIR


