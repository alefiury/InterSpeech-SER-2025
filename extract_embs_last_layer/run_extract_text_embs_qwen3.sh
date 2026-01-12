GPU_ID=4

# CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_text_embs_qwen3.py \
#     --input-csv /hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/Dataset/transcribed_canary_train_set.csv \
#     --output-dir /hadatasets/alef.ferreira/SER/Interspeech/embeddings_qwen3 \
#     --model-name Qwen/Qwen3-Embedding-0.6B \
#     --file-col FileName \
#     --text-col Transcript \
#     --max-length 512

CUDA_VISIBLE_DEVICES=$GPU_ID python3 extract_text_embs_qwen3.py \
    --input-csv /hadatasets/alef.ferreira/SER/Interspeech/InterSpeech-SER-2025/Dataset/validation_set.csv \
    --output-dir /hadatasets/alef.ferreira/SER/Interspeech/embeddings_qwen3 \
    --model-name Qwen/Qwen3-Embedding-0.6B \
    --file-col FileName \
    --text-col Transcript \
    --max-length 512 \
    --transcript-audio \
    --base-dir /hadatasets/alef.ferreira/SER/Interspeech/Audios