import torch

from extract_embs_last_layer.f0_predictor.RMVPEF0Predictor import RMVPEF0Predictor

def get_f0_predictor(hop_length, sampling_rate, **kargs):
    f0_predictor_object = RMVPEF0Predictor(
        hop_length=hop_length,
        sampling_rate=sampling_rate,
        dtype=torch.float32 ,
        device=kargs["device"],
        threshold=kargs["threshold"]
    )
    return f0_predictor_object
