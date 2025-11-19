# Last Layer Feature Extraction

This script extracts embeddings from audio files using SSL speech models. It allows you to specify the base directory, input directory, output directory, and model name for the extraction process.

## Example Execution

To run the script, use the following command:

```
python3 extract_wavlm_embeddings.py -b=PATH_TO_BASE_DIR -i=NAME_OF_DIR_INSIDE_BASE_DIR -o=NAME_OF_OUTPUT_DIR -m="wavlm-large"
```

## Parameters

- `-b` or `--base-dir` (required):
  Path to the base directory where the input directory is located.

- `-i` or `--input-dir-name` (required):
  Name of the input directory, located inside the base directory. This directory contains the audio files to process.

- `-o` or `--output-dir-name`:
  Name of the output directory where the embeddings will be saved. Default is `output_embeddings`.

- `-m` or `--model-name`:
  Model to use for feature extraction. Available choices (for wavlm in this case, each model has its own set of versions) are:
  - `wavlm-large` (default)
  - `wavlm-base-plus`
  - `wavlm-base-plus-sv`

- `-c` or `--input-csv`:
  Path to a metadata CSV file. This is optional and can be used to provide additional information about the input files.

- `-col` or `--column-name`:
  Name of the column in the CSV file that contains the filenames. Default is `filename`.

## Extraction with XEUS Model

XEUS huggingface repo: https://huggingface.co/espnet/xeus

Download the checkpoint and config:

```
download_ssl_chekpoints.sh
```

Install dependencies:

ATTENTION! Make sure to install the ESPnet package in "master" branch, not in "ssl" like in the repo, "ssl" branch is outdated.

```
pip install 'espnet @ git+https://github.com/wanchichen/espnet.git@master'
```