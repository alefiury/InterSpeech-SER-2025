import pandas as pd
import sys
from collections import Counter
import os
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate

# === Configuration ===

# List of CSV file paths containing model predictions
csv_files = [
    "/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/results_true_test/results_true_test-gjzr2c72-true_test.csv",
    '/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/results_true_test/results_true_test-21kikpvt-true_test.csv',
    '/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/results_true_test/results_true_test-jfvtd3zh-true_test.csv',
    '/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/results_true_test/results_true_test-lk44fmxi-true_test.csv',
    '/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/results_true_test/results_true_test-wbv1vclh-true_test.csv',
    '/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/results_true_test/results_true_test-igqworc8-true_test.csv',
    '/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/results_true_test/results_true_test-vyn8y30l-true_test.csv',
    '/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/results_true_test/results_true_test-e7h6dwna-true_test.csv',
]

# Output CSV file path for ensemble predictions
output_file = '/hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/src/results_true_test/ensemble_predictions-gjzr2c72-21kikpvt-jfvtd3zh-lk44fmxi-wbv1vclh-igqworc8-vyn8y30l-e7h6dwna.csv'

# ======================

def read_csv_files(csv_files):
    dataframes = []
    model_names = []
    for file in csv_files:
        if not os.path.isfile(file):
            print(f"Error: File '{file}' does not exist.")
            sys.exit(1)
        try:
            df = pd.read_csv(file)

            dataframes.append(df)
            model_name = os.path.splitext(os.path.basename(file))[0]
            model_names.append(model_name)
        except Exception as e:
            print(f"Error reading '{file}': {e}")
            sys.exit(1)
    return dataframes, model_names

def validate_dataframes(dataframes):
    # Ensure all dataframes have the same FileNames in the same order
    reference = dataframes[0][['FileName']].copy()
    # for idx, df in enumerate(dataframes[1:], start=2):
    #     if not df['FileName'].equals(reference['FileName']):
    #         print(f"Error: FileNames in dataframe {idx} do not match the reference.")
    #         sys.exit(1)
    #     if not df['target'].equals(reference['target']):
    #         print(f"Error: Targets in dataframe {idx} do not match the reference.")
    #         sys.exit(1)
    return reference

def majority_vote(predictions):
    count = Counter(predictions)
    most_common = count.most_common()
    if len(most_common) == 0:
        return None
    max_count = most_common[0][1]
    # Check for ties
    candidates = [pred for pred, cnt in most_common if cnt == max_count]
    if len(candidates) == 1:
        return candidates[0]
    else:
        # Tie-breaking strategy: choose the first candidate
        return candidates[0]

def perform_ensemble(dataframes, reference):
    # Extract predictions from all dataframes
    preds = [df['pred'] for df in dataframes]
    preds_df = pd.concat(preds, axis=1)
    preds_df.columns = [f'Model_{i+1}' for i in range(len(dataframes))]
    
    # Apply majority voting
    ensemble_preds = preds_df.apply(lambda row: majority_vote(row.tolist()), axis=1)
    
    # Combine with FileName and Target
    ensemble_df = reference.copy()
    ensemble_df['Ensemble_Prediction'] = ensemble_preds
    return ensemble_df

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return accuracy, f1_macro, f1_micro

def evaluate_models(dataframes, model_names, reference, ensemble_preds):
    y_true = reference['target']
    metrics = []

    # Evaluate each model
    for df, name in zip(dataframes, model_names):
        y_pred = df['pred']
        acc, f1_mac, f1_mic = calculate_metrics(y_true, y_pred)
        metrics.append({
            'Model': name,
            'Accuracy': acc,
            'F1 Macro': f1_mac,
            'F1 Micro': f1_mic
        })
    
    # Evaluate ensemble
    acc, f1_mac, f1_mic = calculate_metrics(y_true, ensemble_preds)
    metrics.append({
        'Model': 'Ensemble',
        'Accuracy': acc,
        'F1 Macro': f1_mac,
        'F1 Micro': f1_mic
    })

    return metrics

def print_metrics(metrics):
    # Format metrics for display
    table = []
    headers = ['Model', 'Accuracy', 'F1 Macro', 'F1 Micro']
    for metric in metrics:
        table.append([
            metric['Model'],
            f"{metric['Accuracy']:.4f}",
            f"{metric['F1 Macro']:.4f}",
            f"{metric['F1 Micro']:.4f}"
        ])
    
    print("\n=== Evaluation Metrics ===")
    print(tabulate(table, headers=headers, tablefmt="github"))

def main():
    # Read all CSV files
    dataframes, model_names = read_csv_files(csv_files)
    
    # Validate dataframes
    reference = validate_dataframes(dataframes)
    
    # Perform majority voting ensemble
    ensemble_df = perform_ensemble(dataframes, reference)
    
    # Reorder columns for clarity
    ensemble_df = ensemble_df[['FileName', 'Ensemble_Prediction']]
    # ensemble_df.rename(columns={'target': 'EmoClass'}, inplace=True)
    
    print(ensemble_df)

    label2id = {
        "A": 0,
        "C": 1,
        "D": 2,
        "F": 3,
        "H": 4,
        "N": 5,
        "S": 6,
        "U": 7
    }

    id2label = {}
    for label, id in label2id.items():
        id2label[id] = label

    print(id2label)
    print(label2id)

    # convert "Ensemble_Prediction" to labels
    ensemble_df['EmoClass'] = ensemble_df['Ensemble_Prediction'].map(id2label)

    ensemble_df = ensemble_df[['FileName', 'EmoClass']]

    print(ensemble_df)

    # save ensemble predictions to CSV
    ensemble_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
