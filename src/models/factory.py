from models.modules import SERBaseModel, SEREmbeddingModel, SERDynamicModel


def create_ser_model(
    model_type: str,
    **kwargs
) -> SERBaseModel:
    """
    Factory function to create either a SERDynamicModel or a SEREmbeddingModel.
    model_type: "dynamic" or "embedding"
    kwargs: parameters to be passed to the model constructors
    """
    if model_type == "dynamic":
        return SERDynamicModel(**kwargs)
    elif model_type == "embedding":
        return SEREmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'dynamic' or 'embedding'.")