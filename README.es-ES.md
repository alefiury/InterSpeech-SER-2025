

# InterSpeech-SER-2025

Este repositorio contiene el código del artículo titulado [**Mejora del Reconocimiento de Emociones en el Habla con Fusión Multimodal Basada en Grafos y Características Prosódicas para el Desafío de Reconocimiento de Emociones en el Habla en Condiciones Naturalistas en Interspeech 2025**](https://arxiv.org/abs/2506.02088).

Este trabajo obtuvo el **8º puesto** en el Desafío de Reconocimiento de Emociones en el Habla en Condiciones Naturalistas de Interspeech 2025.

## Prerrequisitos
    - Sistema Operativo: Ubuntu (probado en Ubuntu 22.04.4 LTS)
    - Entorno Conda (probado en Conda versión 24.5.0)
    - Versión de Python: Python 3.10.14
    - Versión del Controlador: 535.161.07 (probado en una A100 80GB) o 535.104.05 (probado en una RTX 5000)
    - Versión de CUDA: 12.1 o superior

## Instalación

Instale las dependencias requeridas utilizando el siguiente comando:

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```


## Número de Capas por Modelo

### Modelos Preentrenados de Audio

- HuBERT Large
    - 25 Capas
    - Dimensión de Entrada: 1024
    - facebook/hubert-large-ls960-ft

- HuBERT XLarge
    - 49 Capas
    - Dimensión de Entrada: 1280
    - facebook/hubert-xlarge-ls960-ft

- MMS 1B
    - 49 Capas
    - Dimensión de Entrada: 1280
    - facebook/mms-1b

- MMS 300M
    - 25 Capas
    - Dimensión de Entrada: 1024
    - facebook/mms-300m

- W2V-BERT 2.0
    - 25 Capas
    - Dimensión de Entrada: 1024
    - facebook/w2v-bert-2.0

- Wav2Vec2 Base 960h
    - 13 Capas
    - Dimensión de Entrada: 768
    - facebook/wav2vec2-base-960h

- Wav2Vec2 Large XLSR-53
    - 25 Capas
    - Dimensión de Entrada: 1024
    - facebook/wav2vec2-large-xlsr-53

- Wav2Vec2 XLS-R 1B
    - 49 Capas
    - Dimensión de Entrada: 1280
    - facebook/wav2vec2-xls-r-1b

- Wav2Vec2 XLS-R 2B
    - 49 Capas
    - Dimensión de Entrada: 1280
    - facebook/wav2vec2-xls-r-2b

- Wav2Vec2 XLS-R 300M
    - 25 Capas
    - Dimensión de Entrada: 1024
    - facebook/wav2vec2-xls-r-300m

- WavLM Base Plus
    - 13 Capas
    - Dimensión de Entrada: 1024
    - microsoft/wavlm-base-plus

- WavLM Large
    - 25 Capas
    - Dimensión de Entrada: 1024
    - microsoft/wavlm-large

- Whisper Tiny
    - 5 Capas
    - Dimensión de Entrada: 384
    - openai/whisper-tiny

- Whisper Small
    - 13 Capas
    - Dimensión de Entrada: 768
    - openai/whisper-small

- Whisper Base
    - 7 Capas
    - Dimensión de Entrada: 512
    - openai/whisper-base

- Whisper Medium
    - 25 Capas
    - Dimensión de Entrada: 1024
    - openai/whisper-medium

- Whisper Large
    - 33 Capas
    - Dimensión de Entrada: 1280
    - openai/whisper-large

- Whisper Large V2
    - 33 Capas
    - Dimensión de Entrada: 1280
    - openai/whisper-large-v2

- Whisper Large V3
    - 33 Capas
    - Dimensión de Entrada: 1280
    - openai/whisper-large-v3


### Modelos Preentrenados de Texto

- BERT Base Uncased
    - 12 Capas
    - Dimensión de Entrada: 768
    - bert-base-uncased

- BERT Large Uncased
    - 24 Capas
    - Dimensión de Entrada: 1024
    - bert-large-uncased

- RoBERTa Base
    - 12 Capas
    - Dimensión de Entrada: 768
    - roberta-base

- RoBERTa Large
    - 24 Capas
    - Dimensión de Entrada: 1024
    - roberta-large

- E5 Base
    - 12 Capas
    - Dimensión de Entrada: 768
    - intfloat/e5-base

- E5 Large
    - 24 Capas
    - Dimensión de Entrada: 1024
    - intfloat/e5-large

- Qwen3 Embedding
    - 28 Capas
    - Dimensión de Entrada: 1024
    - Qwen/Qwen3-Embedding-0.6B
