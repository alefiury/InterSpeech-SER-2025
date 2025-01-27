from models.modules import (
    SERBaseModel,
    SEREmbeddingModel,
    SERDynamicModel,
    SERLastLayerEmbeddingModel,
    SERDynamicAudioTextModel,
    SERXEUSModelTextModel,
    SERXEUSTextModelSpeakerEmb,
    SERDynamicAudioTextModelSpeakerEmb,
    SERDynamicAudioTextModelSpeakerEmbMelSpec,
    SERXEUSTextModelSpeakerEmbMelSpec,
    XEUSModel,
    NESTModel,
    SERDynamicAudioTextMelSpecModel,
    SERLastLayerEmbeddingTextModel,
    SERLastLayerEmbeddingTextMelSpecModel,
    SERBimodalEmbeddingModel,
    SERBimodalEmbeddingF0Model,
    SERBimodalEmbeddingMelSpecModel,
    SERBimodalEmbeddingF0MelSpecModel,
    SERDynamicAudioTextF0MelSpecModel
)


def create_ser_model(
    model_type: str,
    **kwargs
) -> SERBaseModel:
    """
    Factory function to create either a SERDynamicModel or a SEREmbeddingModel.
    model_type: "dynamic" or "embedding"
    kwargs: parameters to be passed to the model constructors
    """
    if model_type.lower() == "dynamic":
        return SERDynamicModel(**kwargs)
    elif model_type.lower() == "dynamic_audio_text":
        return SERDynamicAudioTextModel(**kwargs)
    elif model_type.lower() == "xeus_text":
        return SERXEUSModelTextModel(**kwargs)
    elif model_type.lower() == "dynamic_audio_text_speakeremb":
        return SERDynamicAudioTextModelSpeakerEmb(**kwargs)
    elif model_type.lower() == "xeus_text_speakeremb":
        return SERXEUSTextModelSpeakerEmb(**kwargs)
    elif model_type.lower() == "dynamic_audio_text_speakeremb_melspec":
        return SERDynamicAudioTextModelSpeakerEmbMelSpec(**kwargs)
    elif model_type.lower() == "xeus_text_speakeremb_melspec":
        return SERXEUSTextModelSpeakerEmbMelSpec(**kwargs)
    elif model_type.lower() == "embedding":
        return SEREmbeddingModel(**kwargs)
    elif model_type.lower() == "last_layer_embedding":
        return SERLastLayerEmbeddingModel(**kwargs)
    elif model_type.lower() == "xeus":
        return XEUSModel(**kwargs)
    elif model_type.lower() == "nest":
        return NESTModel(**kwargs)
    elif model_type.lower() == "dynamic_audio_text_melspec":
        return SERDynamicAudioTextMelSpecModel(**kwargs)
    elif model_type.lower() == "last_layer_embedding_text":
        return SERLastLayerEmbeddingTextModel(**kwargs)
    elif model_type.lower() == "last_layer_embedding_text_melspec":
        return SERLastLayerEmbeddingTextMelSpecModel(**kwargs)
    elif model_type.lower() == "bimodal_embedding":
        return SERBimodalEmbeddingModel(**kwargs)
    elif model_type.lower() == "bimodal_embedding_f0":
        return SERBimodalEmbeddingF0Model(**kwargs)
    elif model_type.lower() == "bimodal_embedding_melspec":
        return SERBimodalEmbeddingMelSpecModel(**kwargs)
    elif model_type.lower() == "bimodal_embedding_f0_melspec":
        return SERBimodalEmbeddingF0MelSpecModel(**kwargs)
    elif model_type.lower() == "dynamic_audio_text_f0_melspec":
        return SERDynamicAudioTextF0MelSpecModel(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be \
            'dynamic', 'dynamic_audio_text', 'dynamic_audio_text_speakeremb', \
                'embedding', 'last_layer_embedding', 'xeus', \
                    'xeus_text', 'xeus_text_speakeremb' or 'nest'.")