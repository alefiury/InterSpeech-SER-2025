# InterSpeech-SER-2025

## Prerequisites
    - Operating System: Ubuntu (tested on Ubuntu 22.04.4 LTS)
    - Conda Environment (tested on Conda version 24.5.0)
    - Python Version: Python 3.10.14
    - Driver Version: 535.161.07 (tested on a A100 80GB) or 535.104.05 (tested on a RTX 5000)
    - CUDA Version: 12.1 or higher

## Instalation

Install the required dependencies using the following command:

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## Number of Layers per Model

- HuBERT Large
    - 25 Layers
    - Input Dim: 1024
    - facebook/hubert-large-ls960-ft

- HuBERT XLarge
    - 49 Layers
    - Input Dim: 1280
    - facebook/hubert-xlarge-ls960-ft

- MMS 1B
    - 49 Layers
    - Input Dim: 1280
    - facebook/mms-1b

- MMS 300M
    - 25 Layers
    - Input Dim: 1024
    - facebook/mms-300m

- W2V-BERT 2.0
    - 25 Layers
    - Input Dim: 1024
    - facebook/w2v-bert-2.0

- Wav2Vec2 Base 960h
    - 13 Layers
    - Input Dim: 768
    - facebook/wav2vec2-base-960h

- Wav2Vec2 Large XLSR-53
    - 25 Layers
    - Input Dim: 1024
    - facebook/wav2vec2-large-xlsr-53

- Wav2Vec2 XLS-R 1B
    - 49 Layers
    - Input Dim: 1280
    - facebook/wav2vec2-xls-r-1b

- Wav2Vec2 XLS-R 2B
    - 49 Layers
    - Input Dim: 1280
    - facebook/wav2vec2-xls-r-2b

- Wav2Vec2 XLS-R 300M
    - 25 Layers
    - Input Dim: 1024
    - facebook/wav2vec2-xls-r-300m

- WavLM Base Plus
    - 13 Layers
    - Input Dim: 1024
    - microsoft/wavlm-base-plus

- WavLM Large
    - 25 Layers
    - Input Dim: 1280
    - microsoft/wavlm-large

- Whisper Tiny
    - 5 Layers
    - Input Dim: 384
    - openai/whisper-tiny

- Whisper Small
    - 13 Layers
    - Input Dim: 768
    - openai/whisper-small

- Whisper Base
    - 7 Layers
    - Input Dim: 512
    - openai/whisper-base

- Whisper Medium
    - 25 Layers
    - Input Dim: 1024
    - openai/whisper-medium

- Whisper Large
    - 33 Layers
    - Input Dim: 1280
    - openai/whisper-large

- Whisper Large V2
    - 33 Layers
    - Input Dim: 1280
    - openai/whisper-large-v2

- Whisper Large V3
    - 33 Layers
    - Input Dim: 1280
    - openai/whisper-large-v3