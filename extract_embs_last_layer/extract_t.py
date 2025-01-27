import sys
import os
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModel, AutoTokenizer

# Set hardcoded paths
INPUT_CSV = "/hadatasets/alef.ferreira/SER/Interspeech/canary1b_transcripts.csv"
OUTPUT_DIR = "/hadatasets/alef.ferreira/SER/Interspeech"
LAST_LAYER_OUTPUT_DIR = "Texts_E5_last_layer"
POOLER_OUTPUT_DIR = "Texts_E5_pooler"
MODEL_NAME = "intfloat/e5-large-v2"  # Pre-trained model name or path

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("Warning: CUDA is not available. Using CPU.")

def load_model(model_name: str, device: torch.device):
    """
    Load the pre-trained model and tokenizer.
    
    Args:
        model_name (str): Name or path of the pre-trained model.
        device (torch.device): Computation device (CPU or GPU).
    
    Returns:
        model (AutoModel): Loaded pre-trained model.
        tokenizer (AutoTokenizer): Loaded tokenizer.
    """
    print(f"Loading model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def save_embedding(embedding: torch.Tensor, output_filepath: str):
    """
    Save the embedding tensor to a .pt file.
    
    Args:
        embedding (torch.Tensor): The embedding tensor to save.
        output_filepath (str): Path where the embedding will be saved.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    torch.save(embedding.cpu(), output_filepath)

def process_row(row: pd.Series, output_dir: str, model, tokenizer, device: torch.device):
    """
    Extract and save the embedding for a single transcript.
    
    Args:
        row (pd.Series): A row from the DataFrame containing 'FileName' and 'Transcript'.
        output_dir (str): Directory where embeddings will be saved.
        model (AutoModel): Pre-trained model for embedding extraction.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the model.
        device (torch.device): Computation device.

    Returns:
        str or None: The filename of the saved embedding or None if an error occurred.
    """
    try:
        text = row['Transcript']
        filename = row['FileName']
        output_filename = os.path.basename(filename)[:-4] + ".pt"
        last_layer_output_filepath = os.path.join(output_dir, LAST_LAYER_OUTPUT_DIR, output_filename)
        pooler_output_filepath = os.path.join(output_dir, POOLER_OUTPUT_DIR, output_filename)

        if os.path.exists(last_layer_output_filepath) and os.path.exists(pooler_output_filepath):
            return output_filename

        # Tokenize the text
        inputs = tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state.to("cpu")
            pooler_output = outputs.pooler_output.to("cpu")

        # Save the embedding
        save_embedding(last_hidden_states, last_layer_output_filepath)
        save_embedding(pooler_output, pooler_output_filepath)

        # # Save the embedding
        # save_embedding(embedding, output_filepath)
        return output_filename
    except Exception as e:
        print(f"Error processing {row['FileName']}: {e}")
        return None

def extract_embeddings_parallel(df: pd.DataFrame, output_dir: str, model, tokenizer, device: torch.device, max_workers: int = 4):
    """
    Extract embeddings from texts in the DataFrame in parallel using ThreadPoolExecutor.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        output_dir (str): Directory where embeddings will be saved.
        model (AutoModel): Pre-trained model for embedding extraction.
        tokenizer (AutoTokenizer): Tokenizer corresponding to the model.
        device (torch.device): Computation device.
        max_workers (int): Number of threads to use for parallel processing.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        futures = [
            executor.submit(process_row, row, output_dir, model, tokenizer, device)
            for _, row in df.iterrows()
        ]

        # Use tqdm to display a progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting embeddings"):
            result = future.result()
            # Optionally, handle the result (e.g., log successful processing)
            # For now, we ignore it as tqdm handles the progress
            pass

    # After all tasks are completed, count successful embeddings
    successful = df.shape[0] - futures.count(None)
    print(f"Successfully processed {successful}/{len(df)} rows.")

def main():
    """
    Main function to orchestrate the embedding extraction process.
    """
    print(f"Loading CSV file from '{INPUT_CSV}'...")
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded CSV with {len(df)} rows.")
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        sys.exit(1)

    # Load model and tokenizer
    model, tokenizer = load_model(MODEL_NAME, device)

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory at '{OUTPUT_DIR}'.")

    # Determine the number of workers
    if torch.cuda.is_available():
        # If using GPU, limit the number of threads to prevent GPU contention
        max_workers = 4  # Adjust based on your GPU's capability
        print(f"Using {max_workers} threads for GPU-based processing.")
    else:
        # If using CPU, use a higher number of threads
        max_workers = os.cpu_count() or 4
        print(f"Using {max_workers} threads for CPU-based processing.")

    # Extract embeddings in parallel
    print(f"Starting embedding extraction...")
    extract_embeddings_parallel(
        df=df,
        output_dir=OUTPUT_DIR,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_workers=max_workers
    )

    print("Embedding extraction completed successfully.")

if __name__ == "__main__":
    main()
