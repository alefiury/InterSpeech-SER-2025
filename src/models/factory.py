from models.modules import (
    SERBaseModel,
    SEREmbeddingModel,
    SERDynamicModel,
    XEUSModel,
    NESTModel
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
    elif model_type.lower() == "embedding":
        return SEREmbeddingModel(**kwargs)
    elif model_type.lower() == "xeus":
        return XEUSModel(**kwargs)
    elif model_type.lower() == "nest":
        return NESTModel(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'dynamic', 'embedding', 'xeus' or 'nest'.")