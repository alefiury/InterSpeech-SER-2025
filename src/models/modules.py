from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# Backward compatibility
try:
    from espnet2.tasks.ssl import SSLTask
except ImportError:
    SSLTask = None

# Backward compatibility
try:
    from nemo.collections.asr.models import EncDecDenoiseMaskedTokenPredModel
except ImportError:
    EncDecDenoiseMaskedTokenPredModel = None

from models.ced.ced_finetuning import FineTuneCED


ACTIVATIONS_FUNCS = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "leaky_relu": nn.LeakyReLU(),
}


class MLPBase(nn.Module):
    def __init__(
        self,
        input_size: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        output_size: int = 7,
        dropout: float = 0.1,
        activation_func: str = "relu",
    ) -> None:
        super().__init__()

        # Validate and set the activation function
        activation_func = activation_func.lower()
        if activation_func not in ACTIVATIONS_FUNCS:
            raise ValueError(
                f"Unsupported activation function: {activation_func}. "
                f"Supported activations are: {list(ACTIVATIONS_FUNCS.keys())}"
            )
        self.activation_func = activation_func
        activation_layer = ACTIVATIONS_FUNCS[self.activation_func]

        layers: List[nn.Module] = []
        current_dim = input_size

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation_layer)
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_size))
        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes the weights of linear layers based on the activation function:
        ReLU/GELU: Kaiming (He) initialization with a=0, Leaky ReLU: Kaiming (He) initialization with a=0.01
        Biases are initialized to zeros.
        """
        for module in self.layers:
            if isinstance(module, nn.Linear):
                if self.activation_func in ["relu", "gelu"]:
                    # Kaiming initialization with a=0
                    init.kaiming_uniform_(module.weight, a=0, nonlinearity="relu")
                elif self.activation_func == "leaky_relu":
                    # Kaiming initialization with a=0.01
                    init.kaiming_uniform_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x):
        logits = self.layers(x)
        return logits


class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class AttentiveStatisticsPooling(Pooling):
    """
    AttentiveStatisticsPooling
    Paper: Attentive Statistics Pooling for Deep Speaker Embedding
    https://arxiv.org/pdf/1803.10963.pdf

    Input:
    x: (batch_size, T, feat_dim)
    Output:
    (batch_size, feat_dim*2)
    """
    def __init__(self, input_size):
        super().__init__()
        self._indim = input_size
        self.sap_linear = nn.Linear(input_size, input_size)
        self.attention = nn.Parameter(torch.FloatTensor(input_size, 1))
        torch.nn.init.normal_(self.attention, mean=0, std=1)

    def forward(self, xs):
        """
        Args:
            xs: Input tensor of shape (batch_size, T, feat_dim)
        Returns:
            Pooled features of shape (batch_size, feat_dim*2)
        """
        # Calculate attention weights
        h = torch.tanh(self.sap_linear(xs))
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).unsqueeze(2)

        # Calculate weighted mean
        mu = torch.sum(xs * w, dim=1)
        # Calculate weighted standard deviation
        rh = torch.sqrt((torch.sum((xs**2) * w, dim=1) - mu**2).clamp(min=1e-5))
        # Concatenate mean and standard deviation
        pooled = torch.cat((mu, rh), dim=1)
        return pooled


class MoLGating(nn.Module):
    """
    Mixture-of-Layers router that decides how to weigh the layers.
    Two modes:
      - Full Weighted Sum (use_top_k=False):
          Weighted sum across all layers according to a gating distribution.
      - Top-K Weighted Sum (use_top_k=True):
          Keep only the top-k layers per sample, re-normalize among them, and
          do a weighted sum. Layers not in top-k get zero weight.
    Args:
        num_feature_layers (int): Number of layers (experts) available.
        gate_input_dim (int): Dimension of the input to the gating network.
        gate_hidden_dim (int): Hidden dimension in the gating MLP.
        use_top_k (bool): Whether to pick top_k layers or use all layers.
        top_k (int): Number of layers to pick per example if use_top_k=True.
    """
    def __init__(
        self,
        num_feature_layers: int = 25,
        feature_dim: int = 1024,
        gate_hidden_dim: int = 1024,
        use_top_k: bool = False,
        top_k: int = 1,
    ):
        super().__init__()
        self.top_k = top_k
        self.use_top_k = use_top_k
        self.num_feature_layers = num_feature_layers

        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, num_feature_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape [B, L, T, F].
                B = batch size
                L = number of layers (experts)
                T = sequence length
                F = feature dimension

        Returns:
            If self.use_top_k == True:
                Weighted sum of top-K layers => shape [B, F].
            Else:
                Weighted sum of all L layers => shape [B, F].
        """
        B, NUM_LAYERS, SEQ_LEN, FEAT_DIM = x.shape
        x = x.mean(dim=2)  # [B, NUM_LAYERS, FEAT_DIM]

        att_agg = self.self_attn(x, x, x)[0] # [B, NUM_LAYERS, FEAT_DIM]
        gating_input = att_agg.mean(dim=1)  # [B, FEAT_DIM]

        gate_logits = self.gate_network(gating_input)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Use top-k layers
        if self.use_top_k:
            topk_vals, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)
            # Re-normalization among top-k
            topk_sum = topk_vals.sum(dim=-1, keepdim=True)
            topk_probs = topk_vals / topk_sum  # shape => [B, top_k]
            new_gate = torch.zeros_like(gate_probs)
            new_gate.scatter_(dim=-1, index=topk_idx, src=topk_probs)
            final_probs = new_gate
        else:
            # Use all layers
            final_probs = gate_probs # shape [B, NUM_LAYERS]

        final_probs_expanded = final_probs.unsqueeze(-1).expand(-1, NUM_LAYERS, FEAT_DIM)  # [B, NUM_LAYERS, FEAT_DIM]
        # Weighted sum across NUM_LAYERS
        weighted_sum = (x * final_probs_expanded).sum(dim=1) # shape [B, FEAT_DIM]
        return weighted_sum


class SERBaseModel(nn.Module, ABC):
    """
    Base Model for SER that handles:
    - MLP initialization
    - Layer weight strategy: 'per_layer', 'weighted_sum', 'transformer'
    - Pooling strategy: 'mean' or 'attpool' (attpool only with per_layer)
    """

    def __init__(
        self,
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 7,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        layer_weight_strategy: str = "per_layer", # "per_layer" or "weighted_sum"
        num_feature_layers: int = 25,
        specific_layer_idx: int = -1,
        pooling_strategy: str = "mean", # "mean" or "attpool"
        # routing parameters
        gate_hidden_dim: Optional[int] = 1024,
        use_top_k: Optional[bool] = False,
        top_k: Optional[int] = 3,
    ):
        super().__init__()
        self.mlp = MLPBase(
            input_size=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            output_size=mlp_output_size,
            dropout=mlp_dropout,
            activation_func=mlp_activation_func,
        )

        self.layer_weight_strategy = layer_weight_strategy
        self.num_feature_layers = num_feature_layers
        self.specific_layer_idx = specific_layer_idx
        self.pooling_strategy = pooling_strategy

        # Validate layer_weight_strategy
        if layer_weight_strategy == "weighted_sum" or layer_weight_strategy == "weighted_sum_2":
            self.layer_weights = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(num_feature_layers)]
            )
        elif layer_weight_strategy == "routing":
            self.mol_gating = MoLGating(
                num_feature_layers=num_feature_layers,
                feature_dim=mlp_input_dim,
                gate_hidden_dim=gate_hidden_dim,
                use_top_k=use_top_k,
                top_k=top_k,
            )
        elif layer_weight_strategy == "per_layer":
            if specific_layer_idx < 0:
                specific_layer_idx = num_feature_layers - 1
            self.specific_layer_idx = specific_layer_idx
        else:
            raise ValueError(f"Invalid layer weight strategy: {layer_weight_strategy}, choose 'per_layer' or 'weighted_sum'.")

        if pooling_strategy not in ["mean", "attpool"]:
            raise ValueError(
                f"Invalid pooling strategy: {pooling_strategy}. Choose 'mean' or 'attpool'."
            )

        if layer_weight_strategy == "weighted_sum_2" and pooling_strategy != "mean":
            raise ValueError(
                f"Invalid pooling strategy: {pooling_strategy} for layer weight strategy 'weighted_sum_2'. \
                    Choose 'mean' for weighted_sum_2, because it already returns [B,F] \
                        using mean over T dimension in each layer."
            )

        if pooling_strategy == "attpool":
            # AttentiveStatisticsPooling requires initialization once we know F
            # Which is half of the MLP input dimension because we concatenate mean and std
            # Consider that mlp_input_dim is always equal to the F dimension of the embeddings
            self.attpool = AttentiveStatisticsPooling(input_size=int(mlp_input_dim/2))

    def get_layer_weights(self):
        layer_weights = [w.detach().item() for w in self.layer_weights]
        layer_weights = F.softmax(torch.tensor(layer_weights), dim=0)
        return layer_weights

    def _weighted_sum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Weighted sum over the layers dimension.
        Args:
            x: Input tensor of shape [B, NUM_LAYERS, SEQUENCE_LENGTH, FEATURE_DIM]
        Returns:
            Weighted sum tensor of shape [B, SEQUENCE_LENGTH, FEATURE_DIM]
        """

        B, NUM_LAYERS, SEQ_LEN, FEAT_DIM = x.shape

        layer_weights = torch.stack([w for w in self.layer_weights])
        layer_weights = F.softmax(layer_weights, dim=0)
        layer_weights = layer_weights.view(NUM_LAYERS, 1, 1)

        expanded_weights = layer_weights.expand(B, NUM_LAYERS, SEQ_LEN, FEAT_DIM)
        # Apply weights to the input
        weighted_layers = x * expanded_weights
        # Sum over the layers dimension
        # Shape: [B, SEQ_LEN, FEAT_DIM]
        weighted_sum = weighted_layers.sum(dim=1)

        return weighted_sum

    def _weighted_sum_2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Weighted sum over the layers dimension with sequence mean.
        Args:
            x: Input tensor of shape [B, NUM_LAYERS, SEQUENCE_LENGTH, FEATURE_DIM]
        Returns:
            Weighted sum tensor of shape [B, SEQUENCE_LENGTH, FEATURE_DIM]
        """

        B, NUM_LAYERS, SEQ_LEN, FEAT_DIM = x.shape

        x = x.mean(dim=2)

        layer_weights = torch.stack([w for w in self.layer_weights])
        layer_weights = F.softmax(layer_weights, dim=0)
        layer_weights = layer_weights.view(1, NUM_LAYERS, 1)
        expanded_weights = layer_weights.expand(B, NUM_LAYERS, FEAT_DIM)
        # Apply weights to the input
        weighted_layers = x * expanded_weights
        # Sum over the layers dimension
        # Shape: [B, SEQ_LEN, FEAT_DIM]
        weighted_sum = weighted_layers.sum(dim=1)

        return weighted_sum

    def _specific_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return x[:, layer_idx, :]

    @abstractmethod
    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to get embeddings:
        For dynamic model: returns [B,num_layers+1,T,F]
        For embedding model: returns [B,num_feature_layers,F]
        """
        pass

    @abstractmethod
    def _get_embedding_dim(self) -> int:
        pass

    def _apply_layer_weighting(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        if self.layer_weight_strategy == "per_layer":
            embeddings = self._specific_layer(embeddings, self.specific_layer_idx) # [B,T,F]
        elif self.layer_weight_strategy == "weighted_sum":
            embeddings = self._weighted_sum(embeddings) # [B,T,F]
        elif self.layer_weight_strategy == "weighted_sum_2":
            embeddings = self._weighted_sum_2(embeddings) # [B,F]
        elif self.layer_weight_strategy == "routing":
            embeddings = self.mol_gating(embeddings)
        else:
            raise ValueError(f"Invalid layer weight strategy: {self.layer_weight_strategy}")

        return embeddings

    def _apply_pooling(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling over the time dimension.
        embeddings: [B,T,F]
        mask: [B,T] if attpool selected
        """
        if self.pooling_strategy == "mean":
            # Mean pooling over T
            return embeddings.mean(dim=1)  # [B,F]

        elif self.pooling_strategy == "attpool":
            return self.attpool(embeddings)
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        embeddings = self._get_embeddings(x)
        # Apply layer weighting
        embeddings = self._apply_layer_weighting(embeddings)
        # weighted_sum_2 and routing already returns [B,F], so we skip pooling
        if self.layer_weight_strategy == "weighted_sum_2" or self.layer_weight_strategy == "routing":
            logits_input = embeddings
        else:
            # Apply pooling
            logits_input = self._apply_pooling(embeddings)  # [B,F] for "mean" or [B,2F] for "attpool"
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class SEREmbeddingModel(SERBaseModel):
    """
    Speech Emotion Recognition model that uses pre-extracted embeddings as input.
    This model expects pre-computed features instead of processing raw audio through a backbone.

    Input:
        x: Pre-extracted features tensor of shape [batch_size, num_feature_layers, sequence_length, feature_dim]

    The model supports different layer weighting strategies:
    - 'per_layer': Uses features from a specific layer
    - 'weighted_sum': Learns weights to combine features from all layers

    And different pooling strategies:
    - 'mean': Simple mean pooling over the sequence dimension
    - 'attpool': Attentive Statistics Pooling (only available with 'per_layer' strategy)
    """
    def __init__(
        self,
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 7,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        layer_weight_strategy: str = "per_layer",
        num_feature_layers: int = 25,
        specific_layer_idx: int = -1,
        pooling_strategy: str = "mean",
    ):
        super().__init__(
            mlp_input_dim=mlp_input_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_num_layers=mlp_num_layers,
            mlp_output_size=mlp_output_size,
            mlp_dropout=mlp_dropout,
            mlp_activation_func=mlp_activation_func,
            layer_weight_strategy=layer_weight_strategy,
            num_feature_layers=num_feature_layers,
            specific_layer_idx=specific_layer_idx,
            pooling_strategy=pooling_strategy,
        )

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simply returns the pre-extracted features.

        Args:
            x: Input tensor of shape [batch_size, num_feature_layers, sequence_length, feature_dim]
                These are pre-extracted features, unlike SERDynamicModel which processes raw input
                through a backbone.

        Returns:
            The same tensor, as features are already extracted
        """
        return x

    def _get_embedding_dim(self) -> int:
        """
        Returns the dimension of the input features, which is determined by the first layer
        of the MLP.

        Returns:
            int: The feature dimension
        """
        return self.mlp.layers[0].in_features


class SERLastLayerEmbeddingModel(nn.Module):
    """
    Model that uses the last layer of the embeddings model.

    It does not apply layer weighting strategies.
    """

    def __init__(
        self,
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 7,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        pooling_strategy: str = "mean", # "mean" or "attpool"
    ):
        super().__init__()
        self.mlp = MLPBase(
            input_size=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            output_size=mlp_output_size,
            dropout=mlp_dropout,
            activation_func=mlp_activation_func,
        )

        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in ["mean", "attpool"]:
            raise ValueError(
                f"Invalid pooling strategy: {pooling_strategy}. Choose 'mean' or 'attpool'."
            )

        if pooling_strategy == "attpool":
            # AttentiveStatisticsPooling requires initialization once we know F
            # Which is half of the MLP input dimension because we concatenate mean and std
            # Consider that mlp_input_dim is always equal to the F dimension of the embeddings
            self.attpool = AttentiveStatisticsPooling(input_size=int(mlp_input_dim/2))

    def _apply_pooling(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling over the time dimension.
        embeddings: [B,T,F]
        mask: [B,T] if attpool selected
        """
        if self.pooling_strategy == "mean":
            # Mean pooling over T
            return embeddings.mean(dim=1)  # [B,F]

        elif self.pooling_strategy == "attpool":
            return self.attpool(embeddings)
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Apply pooling
        logits_input = self._apply_pooling(embeddings) # [B, F] or [B,2*F]
        # MLP classification
        logits = self.mlp(logits_input)
        return logits


class SERDynamicModel(SERBaseModel):
    """
    Uses a pretrained backbone (e.g. WavLM).
    """
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        freeze_backbone: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=config)

        # Whisper is an encoder-encoder model, we only need the encoder part
        if "whisper" in model_name.lower():
            self.backbone = self.backbone.encoder

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()
            self.backbone.eval()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**x, output_hidden_states=True)
        else:
            outputs = self.backbone(**x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (layer_0,...,layer_n)
        # [num_layers,B,T,F]
        all_layers = torch.stack(hidden_states)
        # transform to [B,num_layers,T,F]
        all_layers = all_layers.permute(1, 0, 2, 3)
        return all_layers

    def _get_embedding_dim(self) -> int:
        return self.mlp.layers[0].in_features


class XEUSModel(SERBaseModel):
    """
    Uses XEUS as backbone.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/xeus_checkpoint.pth",
        freeze_backbone: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone, _ = SSLTask.build_model_from_file(
            None,
            checkpoint_path,
        )

        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            self._freeze_backbone()
            self.backbone.eval()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _get_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        wavs = x["wavs"]
        wav_lengths = x["wav_lengths"]

        if self.freeze_backbone:
            with torch.no_grad():
                hidden_states = self.backbone.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0] # get only the hidden states
        else:
            hidden_states = self.backbone.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)
        # [num_layers,B,T,F]
        all_layers = torch.stack(hidden_states)
        # transform to [B,num_layers,T,F]
        all_layers = all_layers.permute(1, 0, 2, 3)
        return all_layers

    def _get_embedding_dim(self) -> int:
        return self.mlp.layers[0].in_features


class NESTModel(SERBaseModel):
    """
    Uses NEST as backbone.
    """
    def __init__(
        self,
        model_name: str = "nvidia/ssl_en_nest_xlarge_v1.0",
        freeze_backbone: bool = True,
        **kwargs
    ):
        print("INSIDE NEST MODEL")
        super().__init__(**kwargs)
        nest_model = EncDecDenoiseMaskedTokenPredModel.from_pretrained(model_name=model_name)
        self.backbone = nest_model.encoder
        self.preprocessor = nest_model.preprocessor
        self.hidden_states = []

        # Register forward hooks for all encoder layers
        for layer in self.backbone.layers:
            layer.register_forward_hook(self._hook_fn)

        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            self._freeze_backbone()
            self.backbone.eval()

    def _hook_fn(self, module, input, output):
        """Hook function to capture layer outputs"""
        self.hidden_states.append(output)

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _get_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        wavs = x["wavs"]
        wav_lengths = x["wav_lengths"]
        # Clear previous hidden states
        self.hidden_states = []

        if self.freeze_backbone:
            with torch.no_grad():
                # Extract Mel spectrograms
                processed_signal, processed_signal_length = self.preprocessor(input_signal=wavs, length=wav_lengths)
                # Hidden states will be captured by the forward hook
                _, _ = self.backbone(audio_signal=processed_signal, length=processed_signal_length)
        else:
            # Extract Mel spectrograms
            processed_signal, processed_signal_length = self.preprocessor(input_signal=wavs, length=wav_lengths)
            # Hidden states will be captured by the forward hook
            _, _ = self.backbone(audio_signal=processed_signal, length=processed_signal_length)
        # [num_layers,B,T,F]
        all_layers = torch.stack(self.hidden_states)
        # transform to [B,num_layers,T,F]
        all_layers = all_layers.permute(1, 0, 2, 3)
        return all_layers

    def _get_embedding_dim(self) -> int:
        return self.mlp.layers[0].in_features


class SERDynamicAudioTextModel(nn.Module):
    """
    Model that uses audio and text backbones to extract embeddings and combines them using an MLP.

    The model supports different layer weighting strategies:

    - 'per_layer': Uses features from a specific layer
    - 'weighted_sum': Learns weights to combine features from all layers

    And different pooling strategies:

    - 'mean': Simple mean pooling over the sequence dimension
    - 'attpool': Attentive Statistics Pooling (only available with 'per_layer' strategy)
    """
    def __init__(
        self,
        # audio
        audio_model_name: str = "microsoft/wavlm-large",
        freeze_audio_backbone: bool = True,
        audio_layer_weight_strategy: str = "per_layer",
        audio_num_feature_layers: int = 25,
        specific_audio_layer_idx: int = -1,
        audio_pooling_strategy: str = "mean",
        # text
        text_model_name: str = "intfloat/e5-large-v2",
        freeze_text_backbone: bool = True,
        text_layer_weight_strategy: str = "per_layer",
        text_num_feature_layers: int = 25,
        specific_text_layer_idx: int = -1,
        text_pooling_strategy: str = "mean",
        # mlp
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 7,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        # projection
        audio_feat_dim: int = 1024,
        text_feat_dim: int = 1024,
        audio_proj_dropout: float = 0.2,
        text_proj_dropout: float = 0.2,
    ):
        super().__init__()
        audio_config = AutoConfig.from_pretrained(audio_model_name, output_hidden_states=True)
        self.audio_backbone = AutoModel.from_pretrained(audio_model_name, config=audio_config)

        text_config = AutoConfig.from_pretrained(text_model_name, output_hidden_states=True)
        self.text_backbone = AutoModel.from_pretrained(text_model_name, config=text_config)

        # Whisper is an encoder-encoder model, we only need the encoder part
        if "whisper" in audio_model_name.lower():
            self.backbone = self.backbone.encoder

        self.freeze_audio_backbone = freeze_audio_backbone
        self.freeze_text_backbone = freeze_text_backbone
        if freeze_audio_backbone:
            self._freeze_backbone(self.audio_backbone)
            self.audio_backbone.eval()
        if freeze_text_backbone:
            self._freeze_backbone(self.text_backbone)
            self.text_backbone.eval()

        self.audio_layer_weight_strategy = audio_layer_weight_strategy
        self.text_layer_weight_strategy = text_layer_weight_strategy

        self.audio_pooling_strategy = audio_pooling_strategy
        self.text_pooling_strategy = text_pooling_strategy

        # Validate layer_weight_strategy
        if audio_layer_weight_strategy == "weighted_sum":
            self.audio_layer_weights = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(audio_num_feature_layers)]
            )
        elif audio_layer_weight_strategy == "per_layer":
            if specific_audio_layer_idx < 0:
                specific_audio_layer_idx = audio_num_feature_layers - 1
            self.specific_audio_layer_idx = specific_audio_layer_idx

        if text_layer_weight_strategy == "weighted_sum":
            # self.text_layer_weights = nn.ParameterList(
            #     [nn.Parameter(torch.zeros(1)) for _ in range(text_num_feature_layers)]
            # )
            self.text_layer_weights = nn.ParameterList(
                [nn.Parameter(torch.randn(1)) for _ in range(text_num_feature_layers)]
            )
        elif text_layer_weight_strategy == "per_layer":
            if specific_text_layer_idx < 0:
                specific_text_layer_idx = text_num_feature_layers - 1
            self.specific_text_layer_idx = specific_text_layer_idx

        if audio_pooling_strategy == "attpool" and audio_layer_weight_strategy == "per_layer":
            # AttentiveStatisticsPooling requires initialization once we know F
            # Which is half of the MLP input dimension because we concatenate mean and std
            # Consider that mlp_input_dim is always equal to the F dimension of the embeddings
            self.audio_attpool = AttentiveStatisticsPooling(input_size=int(mlp_input_dim/2))

        elif text_pooling_strategy == "attpool" and text_layer_weight_strategy == "per_layer":
            # AttentiveStatisticsPooling requires initialization once we know F
            # Which is half of the MLP input dimension because we concatenate mean and std
            # Consider that mlp_input_dim is always equal to the F dimension of the embeddings
            self.text_attpool = AttentiveStatisticsPooling(input_size=int(mlp_input_dim/2))

        # Audio Projection layer
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feat_dim, audio_feat_dim),
            nn.ReLU(),
            nn.Dropout(audio_proj_dropout),
        )

        # Text Projection layer
        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, text_feat_dim),
            nn.ReLU(),
            nn.Dropout(text_proj_dropout),
        )

        # MLP
        self.mlp = MLPBase(
            input_size=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            output_size=mlp_output_size,
            dropout=mlp_dropout,
            activation_func=mlp_activation_func,
        )

    def _freeze_backbone(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _stack_embeddings(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Stack hidden states of the audio and text backbones.
        Args:
            hidden_states: List of hidden states of the audio and text backbones
        Returns:
            Stacked hidden states of shape [B, num_layers, T, F]
        """
        all_layers = torch.stack(hidden_states)
        # transform to [B,num_layers,T,F]
        all_layers = all_layers.permute(1, 0, 2, 3)
        return all_layers

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        audio, text = x
        if self.freeze_audio_backbone:
            with torch.no_grad():
                audio_outputs = self.audio_backbone(**audio, output_hidden_states=True)
        else:
            audio_outputs = self.audio_backbone(**audio, output_hidden_states=True)

        if self.freeze_text_backbone:
            with torch.no_grad():
                text_outputs = self.text_backbone(**text, output_hidden_states=True)
        else:
            text_outputs = self.text_backbone(**text, output_hidden_states=True)
        audio_hidden_states = self._stack_embeddings(audio_outputs.hidden_states)
        text_hidden_states = self._stack_embeddings(text_outputs.hidden_states)

        return audio_hidden_states, text_hidden_states

    def _specific_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return x[:, layer_idx, :]  # [B,T,F]

    def _weighted_sum(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Weighted sum over the layers dimension.
        Args:
            x: Input tensor of shape [B, NUM_LAYERS, SEQUENCE_LENGTH, FEATURE_DIM]
        Returns:
            Weighted sum tensor of shape [B, SEQUENCE_LENGTH, FEATURE_DIM]
        """

        B, NUM_LAYERS, SEQ_LEN, FEAT_DIM = x.shape
        # take the mean of the hidden states over the layers
        x = x.mean(dim=2)
        if modality == "audio":
            layer_weights = torch.stack([w for w in self.audio_layer_weights])
        elif modality == "text":
            layer_weights = torch.stack([w for w in self.text_layer_weights])
        layer_weights = F.softmax(layer_weights, dim=0)
        layer_weights = layer_weights.view(1, NUM_LAYERS, 1)
        expanded_weights = layer_weights.expand(B, NUM_LAYERS, FEAT_DIM)
        # Apply weights to the input
        weighted_layers = x * expanded_weights
        # Sum over the layers dimension
        # Shape: [B, NUM_LAYERS, FEAT_DIM]
        weighted_sum = weighted_layers.sum(dim=1)
        # Shape: [B, FEAT_DIM]
        return weighted_sum

    def _apply_pooling(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        # Audio Modality
        if self.audio_layer_weight_strategy == "per_layer":
            audio_embeddings = self._specific_layer(audio_hidden_states, self.specific_audio_layer_idx) # [B,T,F]
            # Attentive Statistics Pooling applied only to the per_layer strategy
            if self.audio_pooling_strategy == "attpool":
                audio_embeddings = self.audio_attpool(audio_embeddings)
            else:
                audio_embeddings = audio_embeddings.mean(dim=1)

        elif self.audio_layer_weight_strategy == "weighted_sum":
            audio_embeddings = self._weighted_sum(audio_hidden_states, modality="audio") # [B,T,F]

        # Text Modality
        if self.text_layer_weight_strategy == "per_layer":
            text_embeddings = self._specific_layer(text_hidden_states, self.specific_text_layer_idx)
            # Attentive Statistics Pooling applied only to the per_layer strategy
            if self.text_pooling_strategy == "attpool":
                text_embeddings = self.text_attpool(text_embeddings)
            else:
                text_embeddings = text_embeddings.mean(dim=1)

        elif self.text_layer_weight_strategy == "weighted_sum":
            text_embeddings = self._weighted_sum(text_hidden_states, modality="text")

        return audio_embeddings, text_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        audio_hidden_states, text_hidden_states = self._get_embeddings(x)
        # Apply layer weighting
        audio_embeddings, text_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states)
        # Audio projection
        audio_embeddings = self.audio_proj(audio_embeddings)
        # Text projection
        text_embeddings = self.text_proj(text_embeddings)
        # Concatenate audio and text embeddings
        logits_input = torch.cat((audio_embeddings, text_embeddings), dim=-1)  # [B,AF+TF]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class SERDynamicAudioTextModelSpeakerEmb(SERDynamicAudioTextModel):
    def __init__(
        self,
        # projection
        speaker_emb_dim: int = 512,
        speaker_emb_projection_dim: int = 1024,
        audio_feat_dim: int = 1024,
        text_feat_dim: int = 1024,
        audio_proj_dropout: float = 0.2,
        text_proj_dropout: float = 0.2,
        speaker_emb_dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.speaker_emb_dim = speaker_emb_dim

        # Speaker embedding
        self.speaker_emb_proj = nn.Sequential(
            nn.Linear(speaker_emb_dim, speaker_emb_projection_dim),
            nn.ReLU(),
            nn.Dropout(speaker_emb_dropout),
        )

        # Audio projection
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feat_dim, audio_feat_dim),
            nn.ReLU(),
            nn.Dropout(audio_proj_dropout),
        )

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, text_feat_dim),
            nn.ReLU(),
            nn.Dropout(text_proj_dropout),
        )

    def _get_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio, text, speaker_emb = x
        if self.freeze_audio_backbone:
            with torch.no_grad():
                audio_outputs = self.audio_backbone(**audio, output_hidden_states=True)
        else:
            audio_outputs = self.audio_backbone(**audio, output_hidden_states=True)

        if self.freeze_text_backbone:
            with torch.no_grad():
                text_outputs = self.text_backbone(**text, output_hidden_states=True)
        else:
            text_outputs = self.text_backbone(**text, output_hidden_states=True)
        audio_hidden_states = self._stack_embeddings(audio_outputs.hidden_states)
        text_hidden_states = self._stack_embeddings(text_outputs.hidden_states)

        return audio_hidden_states, text_hidden_states, speaker_emb

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio_hidden_states, text_hidden_states, speaker_emb = self._get_embeddings(x)
        # Apply layer weighting
        audio_embeddings, text_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states)
        # Audio projection
        audio_embeddings = self.audio_proj(audio_embeddings)
        # Text projection
        text_embeddings = self.text_proj(text_embeddings)
        # Speaker embedding projection
        speaker_emb = self.speaker_emb_proj(speaker_emb)
        # Concatenate audio and text embeddings
        logits_input = torch.cat((audio_embeddings, text_embeddings, speaker_emb), dim=-1)
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class SERDynamicAudioTextModelSpeakerEmbMelSpec(SERDynamicAudioTextModel):
    def __init__(
        self,
        # Speaker emb projection
        speaker_emb_dim: int = 512,
        speaker_emb_projection_dim: int = 1024,
        # Audio projection
        audio_feat_dim: int = 1024,
        audio_proj_dropout: float = 0.2,
        # Text projection
        text_feat_dim: int = 1024,
        text_proj_dropout: float = 0.2,
        speaker_emb_dropout: float = 0.2,
        # MelSpec encoder
        mel_spec_encoder_pretrained: bool = True,
        mel_spec_encoder_embedding_dim: int = 768,
        mel_spec_encoder_proj_size: int = 512,
        mel_spec_encoder_proj_dropout: float = 0.2,
        mel_spec_encoder_freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.speaker_emb_dim = speaker_emb_dim

        # Speaker embedding
        self.mel_spec_encoder = FineTuneCED(
            pretrained=mel_spec_encoder_pretrained,
            embedding_dim=mel_spec_encoder_embedding_dim,
            proj_size=mel_spec_encoder_proj_size,
            proj_dropout=mel_spec_encoder_proj_dropout,
            freeze_backbone_flag=mel_spec_encoder_freeze_backbone,
        )

        # Speaker embedding
        self.speaker_emb_proj = nn.Sequential(
            nn.Linear(speaker_emb_dim, speaker_emb_projection_dim),
            nn.ReLU(),
            nn.Dropout(speaker_emb_dropout),
        )

        # Audio projection
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feat_dim, audio_feat_dim),
            nn.ReLU(),
            nn.Dropout(audio_proj_dropout),
        )

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, text_feat_dim),
            nn.ReLU(),
            nn.Dropout(text_proj_dropout),
        )

    def _get_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio, text, speaker_emb = x
        if self.freeze_audio_backbone:
            with torch.no_grad():
                audio_outputs = self.audio_backbone(**audio, output_hidden_states=True)
        else:
            audio_outputs = self.audio_backbone(**audio, output_hidden_states=True)

        if self.freeze_text_backbone:
            with torch.no_grad():
                text_outputs = self.text_backbone(**text, output_hidden_states=True)
        else:
            text_outputs = self.text_backbone(**text, output_hidden_states=True)
        audio_hidden_states = self._stack_embeddings(audio_outputs.hidden_states)
        text_hidden_states = self._stack_embeddings(text_outputs.hidden_states)

        return audio_hidden_states, text_hidden_states, speaker_emb

    def _get_mel_spec_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio, _, _ = x
        audio = audio["input_values"]
        mel_spec_embeddings = self.mel_spec_encoder(audio)
        return mel_spec_embeddings

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio_hidden_states, text_hidden_states, speaker_emb = self._get_embeddings(x)
        # Apply layer weighting
        audio_embeddings, text_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states)
        # Audio projection
        audio_embeddings = self.audio_proj(audio_embeddings)
        # Text projection
        text_embeddings = self.text_proj(text_embeddings)
        # Speaker embedding projection
        speaker_emb = self.speaker_emb_proj(speaker_emb)
        # Mel Spec embeddings
        mel_spec_embeddings = self._get_mel_spec_embeddings(x)
        # Concatenate audio and text embeddings
        logits_input = torch.cat((audio_embeddings, text_embeddings, speaker_emb, mel_spec_embeddings), dim=-1)
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits