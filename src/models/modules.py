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


def swish(x: torch.Tensor) -> torch.Tensor:
    """
    Swish_{β=1}(x) = x * sigmoid(x),
    following Ramachandran et al. (2017).
    """
    return x * torch.sigmoid(x)


class MLP_SwiGLU(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        hidden_dim: int = 512,
        output_size: int = 8,
        dropout: float = 0.1
    ):
        """
        Implements one layer of: (Swish(xW) * (xV)) * W2
        """
        super().__init__()

        # Two "parallel" linear transforms: W and V
        self.W = nn.Linear(input_size, hidden_dim)
        self.V = nn.Linear(input_size, hidden_dim)
        # Final linear transform W2
        self.W2 = nn.Linear(hidden_dim, output_size)
        # Dropout (optional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFN_SwiGLU(x) = (Swish(xW) * xV) * W2
        """
        w_out = self.W(x)
        v_out = self.V(x)
        gated = swish(w_out) * v_out
        gated = self.dropout(gated)
        out = self.W2(gated)

        return out


class MLPBase(nn.Module):
    def __init__(
        self,
        input_size: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        output_size: int = 8,
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
        # gender information
        use_gender_emb: bool = False,
        gender_embedding_dim: int = 16,
    ):
        super().__init__()
        self.gender_encoder = None
        if use_gender_emb:
            print("Using Gender Embedding!")
            self.gender_encoder = nn.Embedding(num_embeddings=2, embedding_dim=gender_embedding_dim)
            mlp_input_dim = mlp_input_dim + gender_embedding_dim

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
        embeddings, genders = self._get_embeddings(x)
        # Apply layer weighting
        embeddings = self._apply_layer_weighting(embeddings)
        # weighted_sum_2 and routing already returns [B,F], so we skip pooling
        if self.layer_weight_strategy == "weighted_sum_2" or self.layer_weight_strategy == "routing":
            logits_input = embeddings
        else:
            # Apply pooling
            logits_input = self._apply_pooling(embeddings)  # [B,F] for "mean" or [B,2F] for "attpool"
        if self.gender_encoder is not None:
            gender_emb = self.gender_encoder(genders)
            logits_input = torch.cat((logits_input, gender_emb), dim=-1)  # [B,AF+TF]
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
        audio_input, genders = x
        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**audio_input, output_hidden_states=True)
        else:
            outputs = self.backbone(**audio_input, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (layer_0,...,layer_n)
        # [num_layers,B,T,F]
        all_layers = torch.stack(hidden_states)
        # transform to [B,num_layers,T,F]
        all_layers = all_layers.permute(1, 0, 2, 3)
        return all_layers, genders

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
        x, genders = x
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
        return all_layers, genders

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
        x, genders = x
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
        return all_layers, genders

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
        # gender information
        use_gender_emb: bool = False,
        gender_embedding_dim: int = 16,
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
        # use transformer encoder
        use_transformer_enc: bool = False,
    ):
        super().__init__()
        self.audio_model_name = audio_model_name
        self.text_model_name = text_model_name

        audio_config = AutoConfig.from_pretrained(audio_model_name, output_hidden_states=True)
        self.audio_backbone = AutoModel.from_pretrained(audio_model_name, config=audio_config)

        text_config = AutoConfig.from_pretrained(text_model_name, output_hidden_states=True)
        self.text_backbone = AutoModel.from_pretrained(text_model_name, config=text_config)

        self.gender_encoder = None
        if use_gender_emb:
            print("Using Gender Embedding!")
            self.gender_encoder = nn.Embedding(num_embeddings=2, embedding_dim=gender_embedding_dim)
            mlp_input_dim = mlp_input_dim + gender_embedding_dim

        # Whisper is an encoder-encoder model, we only need the encoder part
        if "whisper" in audio_model_name.lower():
            self.audio_backbone = self.audio_backbone.encoder

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

        self.use_transformer_enc = use_transformer_enc
        if use_transformer_enc:
            print("Using Transformer Encoder!")
            self.audio_trans_enc = nn.TransformerEncoderLayer(
                d_model=audio_feat_dim,
                nhead=1,
                dim_feedforward=audio_feat_dim*4,
                dropout=audio_proj_dropout,
                batch_first=True
            )

            self.text_trans_enc = nn.TransformerEncoderLayer(
                d_model=text_feat_dim,
                nhead=1,
                dim_feedforward=text_feat_dim*4,
                dropout=text_proj_dropout,
                batch_first=True
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
        audio, text, genders = x
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

        return audio_hidden_states, text_hidden_states, genders

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


    def apply_transformer_enc(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        # Audio Modality
        if self.audio_layer_weight_strategy == "per_layer":
            audio_hidden_states = self._specific_layer(audio_hidden_states, self.specific_audio_layer_idx) # [B,T,F]
            # Attentive Statistics Pooling applied only to the per_layer strategy
        elif self.audio_layer_weight_strategy == "weighted_sum":
            audio_hidden_states = self._weighted_sum(audio_hidden_states, modality="audio") # [B,T,F]

        # Text Modality
        if self.text_layer_weight_strategy == "per_layer":
            text_hidden_states = self._specific_layer(text_hidden_states, self.specific_text_layer_idx)
            # Attentive Statistics Pooling applied only to the per_layer strategy
        elif self.text_layer_weight_strategy == "weighted_sum":
            text_hidden_states = self._weighted_sum(text_hidden_states, modality="text")

        # Projection
        audio_hidden_states = self.audio_proj(audio_hidden_states)
        text_hidden_states = self.text_proj(text_hidden_states)

        # Transformer Encoder
        audio_hidden_states = self.audio_trans_enc(audio_hidden_states)
        text_hidden_states = self.text_trans_enc(text_hidden_states)

        # Mean pooling over T
        audio_embeddings = audio_hidden_states.mean(dim=1)
        text_embeddings = text_hidden_states.mean(dim=1)

        return audio_embeddings, text_embeddings


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        audio_hidden_states, text_hidden_states, genders = self._get_embeddings(x)
        # check if model is using transformers base pooling
        if self.use_transformer_enc:
            audio_embeddings, text_embeddings = self.apply_transformer_enc(audio_hidden_states, text_hidden_states)
        else:
            # Apply layer weighting
            audio_embeddings, text_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states)
            # Audio projection
            audio_embeddings = self.audio_proj(audio_embeddings)
            # Text projection
            text_embeddings = self.text_proj(text_embeddings)
        # Gender embedding
        if self.gender_encoder is not None:
            gender_emb = self.gender_encoder(genders)
            logits_input = torch.cat((audio_embeddings, text_embeddings, gender_emb), dim=-1)  # [B,AF+TF]
        else:
            # Concatenate audio and text embeddings
            logits_input = torch.cat((audio_embeddings, text_embeddings), dim=-1)  # [B, AF+TF]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class SERXEUSModelTextModel(nn.Module):
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
        audio_checkpoint_path: str = None,
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
        self.audio_backbone, _ = SSLTask.build_model_from_file(
            None,
            audio_checkpoint_path,
        )
        text_config = AutoConfig.from_pretrained(text_model_name, output_hidden_states=True)
        self.text_backbone = AutoModel.from_pretrained(text_model_name, config=text_config)

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
        wavs = audio["wavs"]
        wav_lengths = audio["wav_lengths"]

        if self.freeze_audio_backbone:
            with torch.no_grad():
                audio_outputs = self.audio_backbone.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0] # get only the hidden states
        else:
            audio_outputs = self.audio_backbone.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0] # get only the hidden states
        if self.freeze_text_backbone:
            with torch.no_grad():
                text_outputs = self.text_backbone(**text, output_hidden_states=True)
        else:
            text_outputs = self.text_backbone(**text, output_hidden_states=True)
        audio_hidden_states = self._stack_embeddings(audio_outputs)
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
        audio, text, speaker_emb, genders = x
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

        return audio_hidden_states, text_hidden_states, speaker_emb, genders

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio_hidden_states, text_hidden_states, speaker_emb, genders = self._get_embeddings(x)
        # Apply layer weighting
        audio_embeddings, text_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states)
        # Audio projection
        audio_embeddings = self.audio_proj(audio_embeddings)
        # Text projection
        text_embeddings = self.text_proj(text_embeddings)
        # Speaker embedding projection
        speaker_emb = self.speaker_emb_proj(speaker_emb)
        # Concatenate audio and text embeddings
        # Gender embedding
        if self.gender_encoder is not None:
            gender_emb = self.gender_encoder(genders)
            logits_input = torch.cat((audio_embeddings, text_embeddings, speaker_emb, gender_emb), dim=-1)  # [B,AF+TF]
        else:
            # Concatenate audio and text embeddings
            logits_input = torch.cat((audio_embeddings, text_embeddings, speaker_emb), dim=-1)  # [B, AF+TF]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class SERXEUSTextModelSpeakerEmb(SERXEUSModelTextModel):
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
        wavs = audio["wavs"]
        wav_lengths = audio["wav_lengths"]

        if self.freeze_audio_backbone:
            with torch.no_grad():
                audio_outputs = self.audio_backbone.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0] # get only the hidden states
        else:
            audio_outputs = self.audio_backbone.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0] # get only the hidden states

        if self.freeze_text_backbone:
            with torch.no_grad():
                text_outputs = self.text_backbone(**text, output_hidden_states=True)
        else:
            text_outputs = self.text_backbone(**text, output_hidden_states=True)
        audio_hidden_states = self._stack_embeddings(audio_outputs)
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
        wavs_flag = False
        audio, text, speaker_emb = x
        # Remove "wavs" in case the model is Whisper
        if "wavs" in audio:
            wavs = audio.pop("wavs")
            wavs_flag = True
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

        if wavs_flag:
            audio["wavs"] = wavs

        return audio_hidden_states, text_hidden_states, speaker_emb

    def _get_mel_spec_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio, _, _ = x
        if "wavs" in audio:
            audio = audio["wavs"]
        # "input_features" is specific for whisper
        elif "input_features" in audio:
            audio = audio["input_features"]
        else:
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


class SERXEUSTextModelSpeakerEmbMelSpec(SERXEUSModelTextModel):
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
        wavs_flag = False
        audio, text, speaker_emb = x
        wavs = audio["wavs"]
        wav_lengths = audio["wav_lengths"]
        # Remove "wavs" in case the model is Whisper
        if "wavs" in audio:
            wavs = audio.pop("wavs")
            wavs_flag = True
        if self.freeze_audio_backbone:
            with torch.no_grad():
                audio_outputs = self.audio_backbone.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0]
        else:
            audio_outputs = self.audio_backbone.encode(wavs, wav_lengths, use_mask=False, use_final_output=False)[0]

        if self.freeze_text_backbone:
            with torch.no_grad():
                text_outputs = self.text_backbone(**text, output_hidden_states=True)
        else:
            text_outputs = self.text_backbone(**text, output_hidden_states=True)
        audio_hidden_states = self._stack_embeddings(audio_outputs)
        text_hidden_states = self._stack_embeddings(text_outputs.hidden_states)

        if wavs_flag:
            audio["wavs"] = wavs

        return audio_hidden_states, text_hidden_states, speaker_emb

    def _get_mel_spec_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio, _, _ = x
        if "wavs" in audio:
            audio = audio["wavs"]
        # "input_features" is specific for whisper
        elif "input_features" in audio:
            audio = audio["input_features"]
        else:
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


class SERDynamicAudioTextMelSpecModel(SERDynamicAudioTextModel):
    def __init__(
        self,
        # Audio projection
        audio_feat_dim: int = 1024,
        audio_proj_dropout: float = 0.2,
        # Text projection
        text_feat_dim: int = 1024,
        text_proj_dropout: float = 0.2,
        # MelSpec encoder
        mel_spec_encoder_pretrained: bool = True,
        mel_spec_encoder_embedding_dim: int = 768,
        mel_spec_encoder_proj_size: int = 512,
        mel_spec_encoder_proj_dropout: float = 0.2,
        mel_spec_encoder_freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # MelSpec embedding
        self.mel_spec_encoder = FineTuneCED(
            pretrained=mel_spec_encoder_pretrained,
            embedding_dim=mel_spec_encoder_embedding_dim,
            proj_size=mel_spec_encoder_proj_size,
            proj_dropout=mel_spec_encoder_proj_dropout,
            freeze_backbone_flag=mel_spec_encoder_freeze_backbone,
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
        wavs_flag = False
        audio, text, genders = x
        # Remove "wavs" in case the model is Whisper
        if "wavs" in audio:
            wavs = audio.pop("wavs")
            wavs_flag = True
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

        if wavs_flag:
            audio["wavs"] = wavs

        return audio_hidden_states, text_hidden_states, genders

    def _get_mel_spec_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio, _, _ = x
        if "wavs" in audio:
            audio = audio["wavs"]
        # "input_features" is specific for whisper
        elif "input_features" in audio:
            audio = audio["input_features"]
        else:
            audio = audio["input_values"]
        mel_spec_embeddings = self.mel_spec_encoder(audio)
        return mel_spec_embeddings

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio_hidden_states, text_hidden_states, genders = self._get_embeddings(x)
        # check if model is using transformers base pooling
        if self.use_transformer_enc:
            audio_embeddings, text_embeddings = self.apply_transformer_enc(audio_hidden_states, text_hidden_states)
        else:
            # Apply layer weighting
            audio_embeddings, text_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states)
            # Audio projection
            audio_embeddings = self.audio_proj(audio_embeddings)
            # Text projection
            text_embeddings = self.text_proj(text_embeddings)
        # Mel Spec embeddings
        mel_spec_embeddings = self._get_mel_spec_embeddings(x)
        # Concatenate audio and text embeddings
        if self.gender_encoder is not None:
            gender_emb = self.gender_encoder(genders)
            logits_input = torch.cat((audio_embeddings, text_embeddings, mel_spec_embeddings, gender_emb), dim=-1)  # [B,AF+TF]
        else:
            # Concatenate audio and text embeddings
            logits_input = torch.cat((audio_embeddings, text_embeddings, mel_spec_embeddings), dim=-1)  # [B,AF+TF]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class SERLastLayerEmbeddingTextModel(nn.Module):
    """
    Model that uses the last layer of the embeddings model.

    It does not apply layer weighting strategies.
    """

    def __init__(
        self,
        audio_pooling_strategy: str = "mean",
        # text
        text_model_name: str = "intfloat/e5-large-v2",
        freeze_text_backbone: bool = True,
        text_layer_weight_strategy: str = "per_layer",
        text_num_feature_layers: int = 25,
        specific_text_layer_idx: int = -1,
        text_pooling_strategy: str = "mean",
        # gender information
        use_gender_emb: bool = False,
        gender_embedding_dim: int = 16,
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
        # use transformer encoder
        use_transformer_enc: bool = False,
        # use bi-lstm cross-attention pooling
        use_bi_lstm_cross_attn: bool = False,
    ):
        super().__init__()
        text_config = AutoConfig.from_pretrained(text_model_name, output_hidden_states=True)
        self.text_backbone = AutoModel.from_pretrained(text_model_name, config=text_config)

        self.gender_encoder = None
        if use_gender_emb:
            print("Using Gender Embedding!")
            self.gender_encoder = nn.Embedding(num_embeddings=2, embedding_dim=gender_embedding_dim)
            mlp_input_dim = mlp_input_dim + gender_embedding_dim

        self.freeze_text_backbone = freeze_text_backbone
        if freeze_text_backbone:
            self._freeze_backbone(self.text_backbone)
            self.text_backbone.eval()

        self.text_layer_weight_strategy = text_layer_weight_strategy

        self.audio_pooling_strategy = audio_pooling_strategy
        self.text_pooling_strategy = text_pooling_strategy

        # Validate layer_weight_strategy
        if text_layer_weight_strategy == "weighted_sum":
            self.text_layer_weights = nn.ParameterList(
                [nn.Parameter(torch.randn(1)) for _ in range(text_num_feature_layers)]
            )
        elif text_layer_weight_strategy == "per_layer":
            if specific_text_layer_idx < 0:
                specific_text_layer_idx = text_num_feature_layers - 1
            self.specific_text_layer_idx = specific_text_layer_idx

        if audio_pooling_strategy == "attpool":
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

        self.use_transformer_enc = use_transformer_enc
        if use_transformer_enc:
            print("Using Transformer Encoder!")
            self.audio_trans_enc = nn.TransformerEncoderLayer(
                d_model=audio_feat_dim,
                nhead=1,
                dim_feedforward=audio_feat_dim*4,
                dropout=audio_proj_dropout,
                batch_first=True
            )

            self.text_trans_enc = nn.TransformerEncoderLayer(
                d_model=text_feat_dim,
                nhead=1,
                dim_feedforward=text_feat_dim*4,
                dropout=text_proj_dropout,
                batch_first=True
            )

        self.use_bi_lstm_cross_attn = use_bi_lstm_cross_attn
        if use_bi_lstm_cross_attn:
            print("Using Bi-LSTM Cross Attention!")
            self.audio_bi_lstm = nn.LSTM(
                input_size=audio_feat_dim,
                hidden_size=audio_feat_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

            self.text_bi_lstm = nn.LSTM(
                input_size=text_feat_dim,
                hidden_size=text_feat_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

            self.audio_attn = nn.MultiheadAttention(
                embed_dim=audio_feat_dim*2,
                num_heads=8,
                dropout=audio_proj_dropout,
                batch_first=True
            )

            self.text_attn = nn.MultiheadAttention(
                embed_dim=text_feat_dim*2,
                num_heads=8,
                dropout=text_proj_dropout,
                batch_first=True,
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
        audio_hidden_states, text, genders = x

        if self.freeze_text_backbone:
            with torch.no_grad():
                text_outputs = self.text_backbone(**text, output_hidden_states=True)
        else:
            text_outputs = self.text_backbone(**text, output_hidden_states=True)

        text_hidden_states = self._stack_embeddings(text_outputs.hidden_states)

        return audio_hidden_states, text_hidden_states, genders

    def _specific_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return x[:, layer_idx, :]  # [B,T,F]

    def _weighted_sum(self, x: torch.Tensor) -> torch.Tensor:
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
        if self.audio_pooling_strategy == "attpool":
            audio_embeddings = self.audio_attpool(audio_hidden_states)
        else:
            audio_embeddings = audio_hidden_states.mean(dim=1)

        # Text Modality
        if self.text_layer_weight_strategy == "per_layer":
            text_embeddings = self._specific_layer(text_hidden_states, self.specific_text_layer_idx)
            # Attentive Statistics Pooling applied only to the per_layer strategy
            if self.text_pooling_strategy == "attpool":
                text_embeddings = self.text_attpool(text_embeddings)
            else:
                text_embeddings = text_embeddings.mean(dim=1)

        elif self.text_layer_weight_strategy == "weighted_sum":
            text_embeddings = self._weighted_sum(text_hidden_states)

        return audio_embeddings, text_embeddings


    def apply_transformer_enc(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        # Text Modality
        if self.text_layer_weight_strategy == "per_layer":
            text_hidden_states = self._specific_layer(text_hidden_states, self.specific_text_layer_idx)
            # Attentive Statistics Pooling applied only to the per_layer strategy
        elif self.text_layer_weight_strategy == "weighted_sum":
            text_hidden_states = self._weighted_sum(text_hidden_states, modality="text")

        # Projection
        audio_hidden_states = self.audio_proj(audio_hidden_states)
        text_hidden_states = self.text_proj(text_hidden_states)

        # Transformer Encoder
        audio_hidden_states = self.audio_trans_enc(audio_hidden_states)
        text_hidden_states = self.text_trans_enc(text_hidden_states)

        # Mean pooling over T
        audio_embeddings = audio_hidden_states.mean(dim=1)
        text_embeddings = text_hidden_states.mean(dim=1)

        return audio_embeddings, text_embeddings


    def apply_bi_lstm_cross_attn(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        # Text Modality
        if self.text_layer_weight_strategy == "per_layer":
            text_hidden_states = self._specific_layer(text_hidden_states, self.specific_text_layer_idx)
            # Attentive Statistics Pooling applied only to the per_layer strategy
        elif self.text_layer_weight_strategy == "weighted_sum":
            text_hidden_states = self._weighted_sum(text_hidden_states, modality="text")

        # Projection
        audio_hidden_states = self.audio_proj(audio_hidden_states)
        text_hidden_states = self.text_proj(text_hidden_states)

        # Bi-LSTM
        audio_hidden_states, _ = self.audio_bi_lstm(audio_hidden_states)
        text_hidden_states, _ = self.text_bi_lstm(text_hidden_states)

        # Multihead Attention
        audio_hidden_states, _ = self.audio_attn(audio_hidden_states, text_hidden_states, text_hidden_states)
        text_hidden_states, _ = self.text_attn(text_hidden_states, audio_hidden_states, audio_hidden_states)

        # Mean pooling over T
        audio_embeddings = audio_hidden_states.mean(dim=1)
        text_embeddings = text_hidden_states.mean(dim=1)

        return audio_embeddings, text_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        audio_hidden_states, text_hidden_states, genders = self._get_embeddings(x)
        # check if model is using transformers base pooling
        if self.use_transformer_enc:
            audio_embeddings, text_embeddings = self.apply_transformer_enc(audio_hidden_states, text_hidden_states)
        elif self.use_bi_lstm_cross_attn:
            audio_embeddings, text_embeddings = self.apply_bi_lstm_cross_attn(audio_hidden_states, text_hidden_states)
        else:
            # Apply layer weighting
            audio_embeddings, text_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states)
            # Audio projection
            audio_embeddings = self.audio_proj(audio_embeddings)
            # Text projection
            text_embeddings = self.text_proj(text_embeddings)
        # Gender embedding
        if self.gender_encoder is not None:
            gender_emb = self.gender_encoder(genders)
            logits_input = torch.cat((audio_embeddings, text_embeddings, gender_emb), dim=-1)  # [B,AF+TF]
        else:
            # Concatenate audio and text embeddings
            logits_input = torch.cat((audio_embeddings, text_embeddings), dim=-1)  # [B,AF+TF]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class SERLastLayerEmbeddingTextMelSpecModel(SERLastLayerEmbeddingTextModel):
    def __init__(
        self,
        # Audio projection
        audio_feat_dim: int = 1024,
        audio_proj_dropout: float = 0.2,
        # Text projection
        text_feat_dim: int = 1024,
        text_proj_dropout: float = 0.2,
        # MelSpec encoder
        mel_spec_encoder_pretrained: bool = True,
        mel_spec_encoder_embedding_dim: int = 768,
        mel_spec_encoder_proj_size: int = 512,
        mel_spec_encoder_proj_dropout: float = 0.2,
        mel_spec_encoder_freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
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

        # MelSpec embedding
        self.mel_spec_encoder = FineTuneCED(
            pretrained=mel_spec_encoder_pretrained,
            embedding_dim=mel_spec_encoder_embedding_dim,
            proj_size=mel_spec_encoder_proj_size,
            proj_dropout=mel_spec_encoder_proj_dropout,
            freeze_backbone_flag=mel_spec_encoder_freeze_backbone,
        )

    def _get_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio_hidden_states, text, genders, audio = x
        if self.freeze_text_backbone:
            with torch.no_grad():
                text_outputs = self.text_backbone(**text, output_hidden_states=True)
        else:
            text_outputs = self.text_backbone(**text, output_hidden_states=True)

        text_hidden_states = self._stack_embeddings(text_outputs.hidden_states)

        return audio_hidden_states, text_hidden_states, genders, audio

    def _get_mel_spec_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _, _, _, audio = x
        mel_spec_embeddings = self.mel_spec_encoder(audio)
        return mel_spec_embeddings

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio_hidden_states, text_hidden_states, genders, _ = self._get_embeddings(x)
        # check if model is using transformers base pooling
        if self.use_transformer_enc:
            audio_embeddings, text_embeddings = self.apply_transformer_enc(audio_hidden_states, text_hidden_states)
        elif self.use_bi_lstm_cross_attn:
            audio_embeddings, text_embeddings = self.apply_bi_lstm_cross_attn(audio_hidden_states, text_hidden_states)
        else:
            # Apply layer weighting
            audio_embeddings, text_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states)
            # Audio projection
            audio_embeddings = self.audio_proj(audio_embeddings)
            # Text projection
            text_embeddings = self.text_proj(text_embeddings)
        # Mel Spec embeddings
        mel_spec_embeddings = self._get_mel_spec_embeddings(x)
        # Concat Features
        if self.gender_encoder is not None:
            # Gender embedding
            gender_emb = self.gender_encoder(genders)
            logits_input = torch.cat((audio_embeddings, text_embeddings, mel_spec_embeddings, gender_emb), dim=-1)  # [B,AF+TF]
        else:
            # Concatenate audio and text embeddings
            logits_input = torch.cat((audio_embeddings, text_embeddings, mel_spec_embeddings), dim=-1)  # [B,AF+TF]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class GraphAttentionLayer(nn.Module):
    """
    A simple single-head Graph Attention layer,
    treating each time-step as a node in a fully-connected “graph.”
    For a multi-head version, simply wrap multiple layers or use nn.MultiheadAttention.
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2*out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        B, T, F = x.shape
        # Transform
        h = self.W(x)  # [B, T, out_features]

        # Compute pairwise attentions
        # Expand to shape [B, T, T, out_features]
        h_i = h.unsqueeze(2).expand(-1, -1, T, -1)
        h_j = h.unsqueeze(1).expand(-1, T, -1, -1)
        # Concatenate along last dim
        a_input = torch.cat([h_i, h_j], dim=-1)  # [B, T, T, 2*out_features]

        # Score
        e = self.leakyrelu(self.a(a_input)).squeeze(-1)  # [B, T, T]

        # Softmax over j dimension
        # alpha = F.softmax(e, dim=-1)
        alpha = torch.nn.functional.softmax(e, dim=-1)
        # Weighted sum
        # shape of alpha: [B, T, T], shape of h: [B, T, out_features]
        out = torch.bmm(alpha, h)  # [B, T, out_features]
        return out


class CoAttentionLayer(nn.Module):
    """
    A simple co-attention layer that aligns audio/text by computing
    cross-attention weights between them, then producing attended features.
    """
    def __init__(self, d_model):
        super().__init__()
        # Project to a common dimension
        self.proj_audio = nn.Linear(d_model, d_model, bias=False)
        self.proj_text  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, xA: torch.Tensor, xT: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        xA: [B, T_a, F]
        xT: [B, T_t, F]
        Returns: co-attended vectors for audio and text
        """
        # Project each
        A = self.proj_audio(xA)  # [B, T_a, F]
        T = self.proj_text(xT)   # [B, T_t, F]

        # Compute attention matrix
        # [B, T_a, T_t]
        affinity = torch.bmm(A, T.transpose(1, 2))

        # Softmax over T_t dimension for audio context,
        # and over T_a dimension for text context
        alpha_a = F.softmax(affinity, dim=-1)           # [B, T_a, T_t]
        alpha_t = F.softmax(affinity.transpose(1, 2), dim=-1)  # [B, T_t, T_a]

        # Weighted sums -> co-attended representations
        # multiply original xA, xT, not the projected A/T
        coA = torch.bmm(alpha_a, xT)   # [B, T_a, F]
        coT = torch.bmm(alpha_t, xA)   # [B, T_t, F]

        return coA, coT



class SERBimodalEmbeddingModel(nn.Module):
    """
    Model that uses the last layer of the embeddings model.

    It does not apply layer weighting strategies.
    """

    def __init__(
        self,
        audio_pooling_strategy: str = "mean",
        # text
        text_pooling_strategy: str = "mean",
        # mlp
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 7,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        # Audio projection
        audio_feat_dim: int = 1024,
        audio_proj_dim: int = 512,
        audio_proj_dropout: float = 0.2,
        # Text projection
        text_feat_dim: int = 1024,
        text_proj_dim: int = 512,
        text_proj_dropout: float = 0.2,
        # use transformer encoder
        use_transformer_enc: bool = False,
        # use bi-lstm cross-attention pooling
        use_bi_lstm_cross_attn: bool = False,
        # use hierarchical cross-attention model
        use_hcam: bool = False,
        # use MDAT
        use_mdat: bool = False,
        # use Multi-Head Attention MDAT
        use_mdat_multihead: bool = False,
        # use SwiGLU MLP
        use_swiglu_mlp: bool = False,
    ):
        super().__init__()
        # Only one special pooling/fusion can be active at once.
        active_fusions = sum([use_transformer_enc, use_bi_lstm_cross_attn, use_hcam, use_mdat])
        if active_fusions > 1:
            raise ValueError(
                "Only one of {transformer, bi‐lstm cross‐attn, HCAM, MDAT} can be True."
            )

        self.audio_pooling_strategy = audio_pooling_strategy
        self.text_pooling_strategy = text_pooling_strategy

        if audio_pooling_strategy == "attpool":
            # AttentiveStatisticsPooling requires initialization once we know F
            # Which is half of the MLP input dimension because we concatenate mean and std
            # Consider that mlp_input_dim is always equal to the F dimension of the embeddings
            self.audio_attpool = AttentiveStatisticsPooling(input_size=int(mlp_input_dim/2))

        elif text_pooling_strategy == "attpool":
            # AttentiveStatisticsPooling requires initialization once we know F
            # Which is half of the MLP input dimension because we concatenate mean and std
            # Consider that mlp_input_dim is always equal to the F dimension of the embeddings
            self.text_attpool = AttentiveStatisticsPooling(input_size=int(mlp_input_dim/2))


        # Audio Projection layer
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feat_dim, audio_proj_dim),
            nn.ReLU(),
            nn.Dropout(audio_proj_dropout),
        )

        # Text Projection layer
        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, text_proj_dim),
            nn.ReLU(),
            nn.Dropout(text_proj_dropout),
        )

        self.use_transformer_enc = use_transformer_enc
        if use_transformer_enc:
            print("Using Transformer Encoder!")
            self.audio_trans_enc = nn.TransformerEncoderLayer(
                d_model=audio_proj_dim,
                nhead=1,
                dim_feedforward=audio_proj_dim*4,
                dropout=audio_proj_dropout,
                batch_first=True
            )

            self.text_trans_enc = nn.TransformerEncoderLayer(
                d_model=text_proj_dim,
                nhead=1,
                dim_feedforward=text_proj_dim*4,
                dropout=text_proj_dropout,
                batch_first=True
            )

        self.use_bi_lstm_cross_attn = use_bi_lstm_cross_attn
        if use_bi_lstm_cross_attn:
            print("Using Bi-LSTM Cross Attention!")
            self.audio_bi_lstm = nn.GRU(
                input_size=audio_proj_dim,
                hidden_size=audio_proj_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

            self.text_bi_lstm = nn.GRU(
                input_size=text_proj_dim,
                hidden_size=text_proj_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

            self.audio_attn = nn.MultiheadAttention(
                embed_dim=audio_proj_dim*2,
                num_heads=8,
                dropout=audio_proj_dropout,
                batch_first=True
            )

            self.audio_simple_att_proj = nn.Linear(audio_proj_dim*2, audio_proj_dim*2)

            self.text_attn = nn.MultiheadAttention(
                embed_dim=text_proj_dim*2,
                num_heads=8,
                dropout=text_proj_dropout,
                batch_first=True,
            )

            self.text_simple_att_proj = nn.Linear(text_proj_dim*2, text_proj_dim*2)

        self.use_hcam = use_hcam
        if use_hcam:
            print("Using Hierarchical Cross Attention Model (HCAM)!")
            self.audio_context_gru = nn.GRU(
                input_size=audio_proj_dim,
                hidden_size=audio_proj_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.text_context_gru = nn.GRU(
                input_size=text_proj_dim,
                hidden_size=text_proj_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            # Self‐attention after contextual GRU
            self.audio_self_attn = nn.MultiheadAttention(
                embed_dim=audio_proj_dim*2,  # because bidirectional
                num_heads=8,
                batch_first=True
            )
            self.text_self_attn = nn.MultiheadAttention(
                embed_dim=text_proj_dim*2,
                num_heads=8,
                batch_first=True
            )
            self.audio_fc_after_self = nn.Linear(audio_proj_dim*2, audio_proj_dim*2)
            self.audio_ln_after_self = nn.LayerNorm(audio_proj_dim*2)

            self.text_fc_after_self = nn.Linear(text_proj_dim*2, text_proj_dim*2)
            self.text_ln_after_self = nn.LayerNorm(text_proj_dim*2)

            # Cross‐attention blocks (audio→text and text→audio)
            self.cross_attn_a2t = nn.MultiheadAttention(
                embed_dim=audio_proj_dim*2,
                num_heads=8,
                batch_first=True
            )
            self.cross_attn_t2a = nn.MultiheadAttention(
                embed_dim=text_proj_dim*2,
                num_heads=8,
                batch_first=True
            )
            # Post cross‐attn transformations
            self.audio_fc_after_cross = nn.Linear(audio_proj_dim*2, audio_proj_dim*2)
            self.text_fc_after_cross = nn.Linear(text_proj_dim*2, text_proj_dim*2)
            self.audio_ln_after_cross = nn.LayerNorm(audio_proj_dim*2)
            self.text_ln_after_cross = nn.LayerNorm(text_proj_dim*2)

            # Projection layers for final pooling
            self.audio_attentive_pooling_proj = nn.Linear(audio_proj_dim*2, audio_proj_dim*2)
            self.text_attentive_pooling_proj = nn.Linear(text_proj_dim*2, text_proj_dim*2)

        self.use_mdat = use_mdat
        if use_mdat:
            print("Using MDAT (Graph-Att + Co-Att + optional Tx-Encoder).")
            self.audio_graph_attn = GraphAttentionLayer(
                in_features=audio_proj_dim,
                out_features=audio_proj_dim,
            )
            self.text_graph_attn = GraphAttentionLayer(
                in_features=text_proj_dim,
                out_features=text_proj_dim,
            )
            # Co-Attention
            self.co_attention = CoAttentionLayer(d_model=audio_proj_dim)
            # Optional final Transformer Encoders (as in the paper)
            self.audio_trans_enc = nn.TransformerEncoderLayer(
                d_model=audio_proj_dim,
                nhead=8,
                dim_feedforward=audio_proj_dim*4,
                dropout=audio_proj_dropout,
                batch_first=True
            )
            self.text_trans_enc = nn.TransformerEncoderLayer(
                d_model=text_proj_dim,
                nhead=8,
                dim_feedforward=text_proj_dim*4,
                dropout=text_proj_dropout,
                batch_first=True
            )

            # Projection layers for final pooling
            self.audio_attentive_pooling_proj = nn.Linear(audio_proj_dim, audio_proj_dim)
            self.text_attentive_pooling_proj = nn.Linear(text_proj_dim, text_proj_dim)

        self.use_mdat_multihead = use_mdat_multihead
        if use_mdat_multihead:
            print("Using MDAT with Multi-Head Cross-Attention!")
            self.audio_graph_attn = GraphAttentionLayer(
                in_features=audio_proj_dim,
                out_features=audio_proj_dim,
            )
            self.text_graph_attn = GraphAttentionLayer(
                in_features=text_proj_dim,
                out_features=text_proj_dim,
            )
            # Multihead cross-attention layers
            self.mda_audio_to_text = nn.MultiheadAttention(
                embed_dim=audio_proj_dim,
                num_heads=8,               # (Adjust as desired)
                dropout=audio_proj_dropout,
                batch_first=True
            )
            self.mda_text_to_audio = nn.MultiheadAttention(
                embed_dim=text_proj_dim,
                num_heads=8,               # (Adjust as desired)
                dropout=text_proj_dropout,
                batch_first=True
            )

            # Optional final Transformer Encoders (as in the paper)
            self.audio_trans_enc = nn.TransformerEncoderLayer(
                d_model=audio_proj_dim,
                nhead=8,
                dim_feedforward=audio_proj_dim*4,
                dropout=audio_proj_dropout,
                batch_first=True
            )
            self.text_trans_enc = nn.TransformerEncoderLayer(
                d_model=text_proj_dim,
                nhead=8,
                dim_feedforward=text_proj_dim*4,
                dropout=text_proj_dropout,
                batch_first=True
            )

            self.audio_fc_after_cross = nn.Linear(audio_proj_dim, audio_proj_dim)
            self.text_fc_after_cross = nn.Linear(text_proj_dim, text_proj_dim)

            self.audio_ln_after_cross = nn.LayerNorm(audio_proj_dim)
            self.text_ln_after_cross = nn.LayerNorm(text_proj_dim)

            # Projection layers for final pooling
            self.audio_attentive_pooling_proj = nn.Linear(audio_proj_dim, audio_proj_dim)
            self.text_attentive_pooling_proj = nn.Linear(text_proj_dim, text_proj_dim)

        if use_swiglu_mlp:
            print("Using SwishGLU MLP!!!!!!!!!!!")
            self.mlp = MLP_SwiGLU(
                input_size=mlp_input_dim,
                hidden_dim=mlp_hidden_dim,
                output_size=mlp_output_size,
                dropout=mlp_dropout,
            )
        else:
            # MLP
            self.mlp = MLPBase(
                input_size=mlp_input_dim,
                hidden_dim=mlp_hidden_dim,
                num_layers=mlp_num_layers,
                output_size=mlp_output_size,
                dropout=mlp_dropout,
                activation_func=mlp_activation_func,
            )

    def _apply_pooling(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        # Audio Modality
        if self.audio_pooling_strategy == "attpool":
            audio_embeddings = self.audio_attpool(audio_hidden_states)
        else:
            audio_embeddings = audio_hidden_states.mean(dim=1)

        # Text Modality
        if self.text_pooling_strategy == "attpool":
            text_embeddings = self.text_attpool(text_hidden_states)
        else:
            text_embeddings = text_hidden_states.mean(dim=1)

        return audio_embeddings, text_embeddings

    def apply_transformer_enc(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        # Projection
        audio_hidden_states = self.audio_proj(audio_hidden_states)
        text_hidden_states = self.text_proj(text_hidden_states)

        # Transformer Encoder
        audio_hidden_states = self.audio_trans_enc(audio_hidden_states)
        text_hidden_states = self.text_trans_enc(text_hidden_states)

        # Mean pooling over T
        audio_embeddings = audio_hidden_states.mean(dim=1)
        text_embeddings = text_hidden_states.mean(dim=1)

        return audio_embeddings, text_embeddings

    def apply_bi_lstm_cross_attn(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        # Projection
        audio_hidden_states = self.audio_proj(audio_hidden_states)
        text_hidden_states = self.text_proj(text_hidden_states)

        # Bi-LSTM
        audio_hidden_states, _ = self.audio_bi_lstm(audio_hidden_states)
        text_hidden_states, _ = self.text_bi_lstm(text_hidden_states)

        # Multihead Attention
        att_audio_hidden_states, _ = self.audio_attn(audio_hidden_states, text_hidden_states, text_hidden_states)
        att_text_hidden_states, _ = self.text_attn(text_hidden_states, audio_hidden_states, audio_hidden_states)

        audio_hidden_states = audio_hidden_states + att_audio_hidden_states
        text_hidden_states = text_hidden_states + att_text_hidden_states

        # Attention pooling
        audio_att = self.audio_simple_att_proj(audio_hidden_states)
        text_att = self.text_simple_att_proj(text_hidden_states)

        audio_weights = F.softmax(audio_att, dim=1)
        text_weights = F.softmax(text_att, dim=1)

        audio_embeddings = torch.sum(audio_weights * audio_hidden_states, dim=1)
        text_embeddings = torch.sum(text_weights * text_hidden_states, dim=1)

        return audio_embeddings, text_embeddings

    def apply_hcam(self, audio_hidden_states, text_hidden_states):
        """
        Implements a simplified “HCAM” pipeline:
          1) Project each modality.
          2) Contextual Bi‐GRU for each (Stage II style).
          3) Self‐attention on each modality.
          4) Cross‐attention between modalities.
          5) Final pooling + optional concatenation into a single embedding.
        """
        # Projection
        A = self.audio_proj(audio_hidden_states)  # [B,T,a_proj]
        T = self.text_proj(text_hidden_states)    # [B,T,t_proj]

        # GRU for contextual information
        A, _ = self.audio_context_gru(A)  # [B,T,2*a_proj]
        T, _ = self.text_context_gru(T)   # [B,T,2*t_proj]

        # Audio Self‐attention
        A_sa, _ = self.audio_self_attn(A, A, A)
        A = A + A_sa
        A_fc = self.audio_fc_after_self(A)
        # Skip connection
        A = A + A_fc
        A = self.audio_ln_after_self(A)

        # Text Self‐attention
        T_sa, _ = self.text_self_attn(T, T, T)
        T = T + T_sa
        T_fc = self.text_fc_after_self(T)
        # Skip connection
        T = T + T_fc
        T = self.text_ln_after_self(T)

        # Audio Cross‐attention
        A_ca, _ = self.cross_attn_a2t(A, T, T)  # cross from text
        A = A + A_ca
        A_fc2 = self.audio_fc_after_cross(A)
        A = A + A_fc2
        A = self.audio_ln_after_cross(A)

        # Text Cross‐attention
        T_ca, _ = self.cross_attn_t2a(T, A, A)
        T = T + T_ca
        T_fc2 = self.text_fc_after_cross(T)
        T = T + T_fc2
        T = self.text_ln_after_cross(T)

        # Attention pooling
        A_weigths = F.softmax(self.audio_attentive_pooling_proj(A), dim=1)
        T_weigths = F.softmax(self.text_attentive_pooling_proj(T), dim=1)

        audio_embeddings = torch.sum(A_weigths * A, dim=1)
        text_embeddings = torch.sum(T_weigths * T, dim=1)

        return audio_embeddings, text_embeddings

    def apply_mdat(
        self,
        audio_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the MDAT pipeline:
          1) Project audio/text
          2) Graph Attention for each modality
          3) Co-Attention across modalities
          4) Transformer Encoder layers (per modality)
          5) Pool the final sequences (e.g. mean)
        """
        # Projection
        A = self.audio_proj(audio_hidden_states)  # [B, Ta, Da]
        T = self.text_proj(text_hidden_states)    # [B, Tt, Dt]

        # Graph attention for each modality
        A_g = self.audio_graph_attn(A)  # [B, Ta, Da]
        T_g = self.text_graph_attn(T)   # [B, Tt, Dt]

        # Co-attention
        coA, coT = self.co_attention(A_g, T_g)  # [B, Ta, Da], [B, Tt, Dt]

        # Skip connections
        A_co = A_g + coA
        T_co = T_g + coT

        # Transformer encoders
        A_enc = self.audio_trans_enc(A_co)  # [B, Ta, Da]
        T_enc = self.text_trans_enc(T_co)   # [B, Tt, Dt]

        # Attentive pooling
        A_weigths = F.softmax(self.audio_attentive_pooling_proj(A_enc), dim=1)
        T_weigths = F.softmax(self.text_attentive_pooling_proj(T_enc), dim=1)

        audio_embeddings = torch.sum(A_weigths * A_enc, dim=1)
        text_embeddings = torch.sum(T_weigths * T_enc, dim=1)

        return audio_embeddings, text_embeddings

    def apply_mdat_multihead(
        self,
        audio_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        # Projection
        A = self.audio_proj(audio_hidden_states)  # [B, Ta, Da]
        T = self.text_proj(text_hidden_states)    # [B, Tt, Dt]

        # Graph Attention (optional)
        A_g = self.audio_graph_attn(A)  # [B, Ta, Da]
        T_g = self.text_graph_attn(T)   # [B, Tt, Dt]

        # Multi-head cross-attention
        #    Audio attends to text
        #    Query=A_g, Key=Value=T_g => "What does audio learn from text?"
        A_co, _ = self.mda_audio_to_text(A_g, T_g, T_g)
        A_co = A_co + A_g  # skip connection
        A_fc = self.audio_fc_after_cross(A_co)
        A_co = A_co + A_fc # skip connection
        A_co = self.audio_ln_after_cross(A_co)

        #    Text attends to audio
        #    Query=T_g, Key=Value=A_g => "What does text learn from audio?"
        T_co, _ = self.mda_text_to_audio(T_g, A_g, A_g)
        T_co = T_co + T_g  # skip connection
        T_fc = self.text_fc_after_cross(T_co)
        T_co = T_co + T_fc # skip connection
        T_co = self.text_ln_after_cross(T_co)

        # Apply a Transformer Encoder Layer to each modality
        A_enc = self.audio_trans_enc(A_co)  # [B, Ta, Da]
        T_enc = self.text_trans_enc(T_co)   # [B, Tt, Dt]

        # Attentive pooling
        #    Make a learnable linear => produce attention weights, softmax, sum
        A_weights = F.softmax(self.audio_attentive_pooling_proj(A_enc), dim=1)
        T_weights = F.softmax(self.text_attentive_pooling_proj(T_enc),  dim=1)

        audio_embeddings = torch.sum(A_weights * A_enc, dim=1)  # [B, Da]
        text_embeddings  = torch.sum(T_weights * T_enc, dim=1)  # [B, Dt]

        return audio_embeddings, text_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        audio_hidden_states, text_hidden_states = x
        # check if model is using transformers base pooling
        if self.use_transformer_enc:
            audio_embeddings, text_embeddings = self.apply_transformer_enc(audio_hidden_states, text_hidden_states)
        elif self.use_bi_lstm_cross_attn:
            audio_embeddings, text_embeddings = self.apply_bi_lstm_cross_attn(audio_hidden_states, text_hidden_states)
        elif self.use_hcam:
            audio_embeddings, text_embeddings = self.apply_hcam(audio_hidden_states, text_hidden_states)
        elif self.use_mdat:
            audio_embeddings, text_embeddings = self.apply_mdat(audio_hidden_states, text_hidden_states)
        elif self.use_mdat_multihead:
            audio_embeddings, text_embeddings = self.apply_mdat_multihead(audio_hidden_states, text_hidden_states)
        else:
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


class SERBimodalEmbeddingMelSpecModel(SERBimodalEmbeddingModel):
    """
    Model that uses the last layer of the embeddings model.

    It does not apply layer weighting strategies.
    """

    def __init__(
        self,
        # MelSpec encoder
        mel_spec_encoder_pretrained: bool = True,
        mel_spec_encoder_embedding_dim: int = 768,
        mel_spec_encoder_proj_size: int = 512,
        mel_spec_encoder_proj_dropout: float = 0.2,
        mel_spec_encoder_freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # MelSpec embedding
        self.mel_spec_encoder = FineTuneCED(
            pretrained=mel_spec_encoder_pretrained,
            embedding_dim=mel_spec_encoder_embedding_dim,
            proj_size=mel_spec_encoder_proj_size,
            proj_dropout=mel_spec_encoder_proj_dropout,
            freeze_backbone_flag=mel_spec_encoder_freeze_backbone,
        )

    def _get_mel_spec_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _, _, audio = x
        mel_spec_embeddings = self.mel_spec_encoder(audio)
        return mel_spec_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        audio_hidden_states, text_hidden_states, _ = x
        # check if model is using transformers base pooling
        if self.use_transformer_enc:
            audio_embeddings, text_embeddings = self.apply_transformer_enc(audio_hidden_states, text_hidden_states)
        elif self.use_bi_lstm_cross_attn:
            audio_embeddings, text_embeddings = self.apply_bi_lstm_cross_attn(audio_hidden_states, text_hidden_states)
        else:
            # Apply layer weighting
            audio_embeddings, text_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states)
            # Audio projection
            audio_embeddings = self.audio_proj(audio_embeddings)
            # Text projection
            text_embeddings = self.text_proj(text_embeddings)
         # Mel Spec embeddings
        mel_spec_embeddings = self._get_mel_spec_embeddings(x)
        # Concatenate audio and text embeddings
        logits_input = torch.cat((audio_embeddings, text_embeddings, mel_spec_embeddings), dim=-1)  # [B,AF+TF]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits



class SERBimodalEmbeddingF0Model(SERBimodalEmbeddingModel):
    """
    Model that uses the last layer of the embeddings model.

    It does not apply layer weighting strategies.
    """

    def __init__(
        self,
        # F0 params
        num_f0_bins: int = 256,
        f0_embedding_dim: int = 256,
        f0_proj_size: int = 512,
        f0_proj_dropout: float = 0.5,
        f0_pooling_strategy: str = "mean",
        use_cross_f0_attn: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # MelSpec embedding
        self.f0_embedding = nn.Embedding(num_f0_bins+1, f0_embedding_dim) # +1 for padding token

        # F0 projection
        self.f0_proj = nn.Sequential(
            nn.Linear(f0_embedding_dim, f0_proj_size),
            nn.ReLU(),
            nn.Dropout(f0_proj_dropout),
        )

        if self.use_transformer_enc:
            self.f0_trans_enc = nn.TransformerEncoderLayer(
                d_model=f0_proj_size,
                nhead=1,
                dim_feedforward=f0_proj_size*4,
                dropout=f0_proj_dropout,
                batch_first=True
            )

        self.use_cross_f0_attn = use_cross_f0_attn
        if self.use_bi_lstm_cross_attn and use_cross_f0_attn:
            print("Using Bi-LSTM Cross Attention for F0!")
            self.f0_bi_lstm = nn.GRU(
                input_size=f0_proj_size,
                hidden_size=f0_proj_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

            self.f0_attn = nn.MultiheadAttention(
                embed_dim=f0_proj_size*2,
                num_heads=8,
                dropout=f0_proj_dropout,
                batch_first=True
            )

            self.f0_simple_att_proj = nn.Linear(f0_proj_size*2, f0_proj_size*2)

        if self.use_hcam and use_cross_f0_attn:
            print("Using Hierarchical Cross Attention Model (HCAM) for F0!")
            self.f0_context_gru = nn.GRU(
                input_size=f0_proj_size,
                hidden_size=f0_proj_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

            self.f0_self_attn = nn.MultiheadAttention(
                embed_dim=f0_proj_size*2,
                num_heads=8,
                batch_first=True
            )

            self.f0_fc_after_self = nn.Linear(f0_proj_size*2, f0_proj_size*2)
            self.f0_ln_after_self = nn.LayerNorm(f0_proj_size*2)

            self.f0_attn = nn.MultiheadAttention(
                embed_dim=f0_proj_size*2,
                num_heads=8,
                batch_first=True
            )

            self.f0_fc_after_cross = nn.Linear(f0_proj_size*2, f0_proj_size*2)
            self.f0_ln_after_cross = nn.LayerNorm(f0_proj_size*2)

            self.f0_attentive_pooling_proj = nn.Linear(f0_proj_size*2, f0_proj_size*2)

        if self.use_mdat and use_cross_f0_attn:
            print("Using MDAT (Graph-Att + Co-Att for F0!)")
            # Graph Attention for f0
            self.f0_graph_attn = GraphAttentionLayer(
                in_features=f0_proj_size,
                out_features=f0_proj_size,
            )

            # Co-Attention for f0
            self.f0_co_attention = CoAttentionLayer(d_model=f0_proj_size)

            # Optional final Transformer Encoders (as in the paper)
            self.f0_trans_enc = nn.TransformerEncoderLayer(
                d_model=f0_proj_size,
                nhead=8,
                dim_feedforward=f0_proj_size*4,
                dropout=f0_proj_dropout,
                batch_first=True
            )

            # Projection layers for final pooling
            self.f0_attentive_pooling_proj = nn.Linear(f0_proj_size, f0_proj_size)

        self.f0_pooling_strategy = f0_pooling_strategy
        if f0_pooling_strategy == "attpool":
            # AttentiveStatisticsPooling requires initialization once we know F
            # Which is half of the MLP input dimension because we concatenate mean and std
            # Consider that mlp_input_dim is always equal to the F dimension of the embeddings
            self.f0_attpool = AttentiveStatisticsPooling(input_size=int(mlp_input_dim/2))

    def _apply_pooling(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor, f0_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        # Audio Modality
        if self.audio_pooling_strategy == "attpool":
            audio_embeddings = self.audio_attpool(audio_hidden_states)
        else:
            audio_embeddings = audio_hidden_states.mean(dim=1)

        # Text Modality
        if self.text_pooling_strategy == "attpool":
            text_embeddings = self.text_attpool(text_hidden_states)
        else:
            text_embeddings = text_hidden_states.mean(dim=1)

        # F0 Modality
        if self.f0_pooling_strategy == "attpool":
            f0_embeddings = self.f0_attpool(f0_hidden_states)
        else:
            f0_embeddings = f0_hidden_states.mean(dim=1)

        return audio_embeddings, text_embeddings, f0_embeddings

    def apply_transformer_enc(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor, f0_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Projection
        audio_hidden_states = self.audio_proj(audio_hidden_states)
        text_hidden_states = self.text_proj(text_hidden_states)
        f0_hidden_states = self.f0_proj(f0_hidden_states)

        # Transformer Encoder
        audio_hidden_states = self.audio_trans_enc(audio_hidden_states)
        text_hidden_states = self.text_trans_enc(text_hidden_states)
        f0_hidden_states = self.f0_trans_enc(f0_hidden_states)

        # Mean pooling over T
        audio_embeddings = audio_hidden_states.mean(dim=1)
        text_embeddings = text_hidden_states.mean(dim=1)
        f0_embeddings = f0_hidden_states.mean(dim=1)

        return audio_embeddings, text_embeddings, f0_embeddings

    def apply_bi_lstm_cross_attn(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor, f0_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Projection
        audio_hidden_states = self.audio_proj(audio_hidden_states)
        text_hidden_states = self.text_proj(text_hidden_states)
        f0_hidden_states = self.f0_proj(f0_hidden_states)

        # Bi-LSTM
        audio_hidden_states, _ = self.audio_bi_lstm(audio_hidden_states)
        text_hidden_states, _ = self.text_bi_lstm(text_hidden_states)

        # Multihead Attention
        att_audio_hidden_states, _ = self.audio_attn(audio_hidden_states, text_hidden_states, text_hidden_states)
        att_text_hidden_states, _ = self.text_attn(text_hidden_states, audio_hidden_states, audio_hidden_states)

        audio_hidden_states = audio_hidden_states + att_audio_hidden_states
        text_hidden_states = text_hidden_states + att_text_hidden_states

        if self.use_cross_f0_attn:
            # Bi-LSTM
            f0_hidden_states, _ = self.f0_bi_lstm(f0_hidden_states)

            # Multihead Attention
            att_f0_hidden_states_audio, _ = self.f0_attn(f0_hidden_states, audio_hidden_states, audio_hidden_states)
            att_f0_hidden_states_text, _ = self.f0_attn(f0_hidden_states, text_hidden_states, text_hidden_states)

            # Attention pooling
            f0_hidden_states = f0_hidden_states + att_f0_hidden_states_audio + att_f0_hidden_states_text
            f0_att = self.f0_simple_att_proj(f0_hidden_states)
            f0_weights = F.softmax(f0_att, dim=1)
            f0_embeddings = torch.sum(f0_weights * f0_hidden_states, dim=1)
        else:
            f0_embeddings = f0_hidden_states.mean(dim=1)

        audio_att = self.audio_simple_att_proj(audio_hidden_states)
        text_att = self.text_simple_att_proj(text_hidden_states)

        audio_weights = F.softmax(audio_att, dim=1)
        text_weights = F.softmax(text_att, dim=1)

        audio_embeddings = torch.sum(audio_weights * audio_hidden_states, dim=1)
        text_embeddings = torch.sum(text_weights * text_hidden_states, dim=1)

        return audio_embeddings, text_embeddings, f0_embeddings

    def apply_hcam(self, audio_hidden_states, text_hidden_states, f0_hidden_states):
        """
        Implements a simplified “HCAM” pipeline:
          1) Project each modality.
          2) Contextual Bi‐GRU for each (Stage II style).
          3) Self‐attention on each modality.
          4) Cross‐attention between modalities.
          5) Final pooling + optional concatenation into a single embedding.
        """
        # Projection
        A = self.audio_proj(audio_hidden_states)  # [B,T,a_proj]
        T = self.text_proj(text_hidden_states)    # [B,T,t_proj]
        F0 = self.f0_proj(f0_hidden_states)       # [B,T,f0_proj]

        # GRU for contextual information
        A, _ = self.audio_context_gru(A)  # [B,T,2*a_proj]
        T, _ = self.text_context_gru(T)   # [B,T,2*t_proj]

        # Audio Self‐attention
        A_sa, _ = self.audio_self_attn(A, A, A)
        A = A + A_sa
        A_fc = self.audio_fc_after_self(A)
        # Skip connection
        A = A + A_fc
        A = self.audio_ln_after_self(A)

        # Text Self‐attention
        T_sa, _ = self.text_self_attn(T, T, T)
        T = T + T_sa
        T_fc = self.text_fc_after_self(T)
        # Skip connection
        T = T + T_fc
        T = self.text_ln_after_self(T)

        # Audio Cross‐attention
        A_ca, _ = self.cross_attn_a2t(A, T, T)  # cross from text
        A = A + A_ca
        A_fc2 = self.audio_fc_after_cross(A)
        A = A + A_fc2
        A = self.audio_ln_after_cross(A)

        # Text Cross‐attention
        T_ca, _ = self.cross_attn_t2a(T, A, A)
        T = T + T_ca
        T_fc2 = self.text_fc_after_cross(T)
        T = T + T_fc2
        T = self.text_ln_after_cross(T)

        # Attention pooling
        A_weigths = F.softmax(self.audio_attentive_pooling_proj(A), dim=1)
        T_weigths = F.softmax(self.text_attentive_pooling_proj(T), dim=1)

        audio_embeddings = torch.sum(A_weigths * A, dim=1)
        text_embeddings = torch.sum(T_weigths * T, dim=1)

        if self.use_cross_f0_attn:
            # GRU for contextual information
            F0, _ = self.f0_context_gru(F0)  # [B,T,2*f0_proj]

            # F0 Self‐attention
            F0_sa, _ = self.f0_self_attn(F0, F0, F0)
            F0 = F0 + F0_sa
            F0_fc = self.f0_fc_after_self(F0)
            # Skip connection
            F0 = F0 + F0_fc
            F0 = self.f0_ln_after_self(F0)

            # F0-Audio Cross‐attention
            F0_ca_A, _ = self.f0_attn(F0, A, A)
            F0_A = F0 + F0_ca_A
            F0_fc_A = self.f0_fc_after_cross(F0_A)
            F0_A = F0_A + F0_fc_A
            F0_A = self.f0_ln_after_cross(F0_A)

            # F0-Text Cross‐attention
            F0_ca_T, _ = self.f0_attn(F0, T, T)
            F0_T = F0 + F0_ca_T
            F0_fc_T = self.f0_fc_after_cross(F0_T)
            F0_T = F0_T + F0_fc_T
            F0_T = self.f0_ln_after_cross(F0_T)

            # Attention pooling
            F0 = F0 + F0_A + F0_T

            # Attention pooling
            F0_weigths = F.softmax(self.f0_attentive_pooling_proj(F0), dim=1)
            f0_embeddings = torch.sum(F0_weigths * F0, dim=1)
        else:
            f0_embeddings = F0.mean(dim=1)

        return audio_embeddings, text_embeddings, f0_embeddings

    def apply_mdat(
        self,
        audio_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
        f0_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the MDAT pipeline:
          1) Project audio/text
          2) Graph Attention for each modality
          3) Co-Attention across modalities
          4) Transformer Encoder layers (per modality)
          5) Pool the final sequences (e.g. mean)
        """
        # Projection
        A = self.audio_proj(audio_hidden_states)  # [B, Ta, Da]
        T = self.text_proj(text_hidden_states)    # [B, Tt, Dt]
        F0 = self.f0_proj(f0_hidden_states)       # [B, Tf, Df]

        # Graph attention for each modality
        A_g = self.audio_graph_attn(A)  # [B, Ta, Da]
        T_g = self.text_graph_attn(T)   # [B, Tt, Dt]

        # Co-attention
        coA, coT = self.co_attention(A_g, T_g)  # [B, Ta, Da], [B, Tt, Dt]

        # Skip connections
        A_co = A_g + coA
        T_co = T_g + coT

        # Transformer encoders
        A_enc = self.audio_trans_enc(A_co)  # [B, Ta, Da]
        T_enc = self.text_trans_enc(T_co)   # [B, Tt, Dt]

        # Attentive pooling
        A_weigths = F.softmax(self.audio_attentive_pooling_proj(A_enc), dim=1)
        T_weigths = F.softmax(self.text_attentive_pooling_proj(T_enc), dim=1)

        audio_embeddings = torch.sum(A_weigths * A_enc, dim=1)
        text_embeddings = torch.sum(T_weigths * T_enc, dim=1)

        if self.use_cross_f0_attn:
            # Graph attention for f0
            F0_g = self.f0_graph_attn(F0)

            # Co-Attention for f0 and audio
            coF0, coA = self.f0_co_attention(F0_g, A_g)
            F0_co_A = F0_g + coF0

            # Co-Attention for f0 and text
            coF0, coT = self.f0_co_attention(F0_g, T_g)
            F0_co_T = F0_g + coF0

            # Skip connections
            F0_co = F0_g + F0_co_A + F0_co_T

            # Transformer encoders
            F0_enc = self.f0_trans_enc(F0_co)  # [B, Tf, Df]

            # Attentive pooling
            F0_weigths = F.softmax(self.f0_attentive_pooling_proj(F0_enc), dim=1)

            f0_embeddings = torch.sum(F0_weigths * F0_enc, dim=1)
        else:
            f0_embeddings = F0.mean(dim=1)

        return audio_embeddings, text_embeddings, f0_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        audio_hidden_states, text_hidden_states, f0 = x
        f0_embeddings = self.f0_embedding(f0)
        # check if model is using transformers base pooling
        if self.use_transformer_enc:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_transformer_enc(audio_hidden_states, text_hidden_states, f0_embeddings)
        elif self.use_bi_lstm_cross_attn:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_bi_lstm_cross_attn(audio_hidden_states, text_hidden_states, f0_embeddings)
        elif self.use_hcam:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_hcam(audio_hidden_states, text_hidden_states, f0_embeddings)
        elif self.use_mdat:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_mdat(audio_hidden_states, text_hidden_states, f0_embeddings)
        else:
            # Apply layer weighting
            audio_embeddings, text_embeddings, f0_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states, f0_embeddings)
            # Audio projection
            audio_embeddings = self.audio_proj(audio_embeddings)
            # Text projection
            text_embeddings = self.text_proj(text_embeddings)
            # F0 projection
            f0_embeddings = self.f0_proj(f0_embeddings)
        # Concatenate audio and text embeddings
        logits_input = torch.cat((audio_embeddings, text_embeddings, f0_embeddings), dim=-1)  # [B,AF+TF]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class SERBimodalEmbeddingF0MelSpecModel(SERBimodalEmbeddingF0Model):
    """
    Model that uses the last layer of the embeddings model.

    It does not apply layer weighting strategies.
    """

    def __init__(
        self,
        # MelSpec encoder
        mel_spec_encoder_pretrained: bool = True,
        mel_spec_encoder_embedding_dim: int = 768,
        mel_spec_encoder_proj_size: int = 512,
        mel_spec_encoder_proj_dropout: float = 0.2,
        mel_spec_encoder_freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # MelSpec embedding
        self.mel_spec_encoder = FineTuneCED(
            pretrained=mel_spec_encoder_pretrained,
            embedding_dim=mel_spec_encoder_embedding_dim,
            proj_size=mel_spec_encoder_proj_size,
            proj_dropout=mel_spec_encoder_proj_dropout,
            freeze_backbone_flag=mel_spec_encoder_freeze_backbone,
        )

    def _get_mel_spec_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        _, _, _, audio = x
        mel_spec_embeddings = self.mel_spec_encoder(audio)
        return mel_spec_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        audio_hidden_states, text_hidden_states, f0, _ = x
        f0_embeddings = self.f0_embedding(f0)
        # check if model is using transformers base pooling
        if self.use_transformer_enc:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_transformer_enc(audio_hidden_states, text_hidden_states, f0_embeddings)
        elif self.use_bi_lstm_cross_attn:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_bi_lstm_cross_attn(audio_hidden_states, text_hidden_states, f0_embeddings)
        elif self.use_hcam:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_hcam(audio_hidden_states, text_hidden_states, f0_embeddings)
        elif self.use_mdat:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_mdat(audio_hidden_states, text_hidden_states, f0_embeddings)
        else:
            # Apply layer weighting
            audio_embeddings, text_embeddings, f0_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states, f0_embeddings)
            # Audio projection
            audio_embeddings = self.audio_proj(audio_embeddings)
            # Text projection
            text_embeddings = self.text_proj(text_embeddings)
            # F0 projection
            f0_embeddings = self.f0_proj(f0_embeddings)
        # Mel Spec embeddings
        mel_spec_embeddings = self._get_mel_spec_embeddings(x)
        # Concatenate audio and text embeddings
        logits_input = torch.cat((audio_embeddings, text_embeddings, f0_embeddings, mel_spec_embeddings), dim=-1) # [B,AF+TF]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits


class SERDynamicAudioTextF0MelSpecModel(SERDynamicAudioTextModel):
    def __init__(
        self,
        # Audio projection
        audio_feat_dim: int = 1024,
        audio_proj_dim: int = 512,
        audio_proj_dropout: float = 0.5,
        # Text projection
        text_feat_dim: int = 1024,
        text_proj_dim: int = 512,
        text_proj_dropout: float = 0.5,
        # F0 params
        num_f0_bins: int = 256,
        f0_embedding_dim: int = 256,
        f0_proj_size: int = 512,
        f0_proj_dropout: float = 0.5,
        f0_pooling_strategy: str = "mean",
        # MelSpec encoder
        mel_spec_encoder_pretrained: bool = True,
        mel_spec_encoder_embedding_dim: int = 768,
        mel_spec_encoder_proj_size: int = 512,
        mel_spec_encoder_proj_dropout: float = 0.5,
        mel_spec_encoder_freeze_backbone: bool = False,
        # Use F0 Cross-Attention
        use_cross_f0_attn: bool = False,
        # use transformer encoder
        use_transformer_enc: bool = False,
        # use bi-lstm cross-attention pooling
        use_bi_lstm_cross_attn: bool = False,
        # use hierarchical cross-attention model
        use_hcam: bool = False,
        # use MDAT
        use_mdat: bool = False,
        # use Multi-Head Attention MDAT
        use_mdat_multihead: bool = False,
        use_mdat_multihead_gru: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Only one special pooling/fusion can be active at once.
        active_fusions = sum([use_transformer_enc, use_bi_lstm_cross_attn, use_hcam, use_mdat])
        if active_fusions > 1:
            raise ValueError(
                "Only one of {transformer, bi‐lstm cross‐attn, HCAM, MDAT} can be True."
            )

        # MelSpec embedding
        self.mel_spec_encoder = FineTuneCED(
            pretrained=mel_spec_encoder_pretrained,
            embedding_dim=mel_spec_encoder_embedding_dim,
            proj_size=mel_spec_encoder_proj_size,
            proj_dropout=mel_spec_encoder_proj_dropout,
            freeze_backbone_flag=mel_spec_encoder_freeze_backbone,
        )

        # Audio Projection layer
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feat_dim, audio_proj_dim),
            nn.ReLU(),
            nn.Dropout(audio_proj_dropout),
        )

        # Text Projection layer
        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, text_proj_dim),
            nn.ReLU(),
            nn.Dropout(text_proj_dropout),
        )

        self.f0_embedding = nn.Embedding(num_f0_bins+1, f0_embedding_dim) # +1 for padding token

        # F0 projection
        self.f0_proj = nn.Sequential(
            nn.Linear(f0_embedding_dim, f0_proj_size),
            nn.ReLU(),
            nn.Dropout(f0_proj_dropout),
        )

        self.use_cross_f0_attn = use_cross_f0_attn

        self.use_transformer_enc = use_transformer_enc
        if use_transformer_enc:
            print("Using Transformer Encoder!")
            self.audio_trans_enc = nn.TransformerEncoderLayer(
                d_model=audio_proj_dim,
                nhead=1,
                dim_feedforward=audio_proj_dim*4,
                dropout=audio_proj_dropout,
                batch_first=True
            )

            self.text_trans_enc = nn.TransformerEncoderLayer(
                d_model=text_proj_dim,
                nhead=1,
                dim_feedforward=text_proj_dim*4,
                dropout=text_proj_dropout,
                batch_first=True
            )

            self.f0_trans_enc = nn.TransformerEncoderLayer(
                d_model=f0_proj_size,
                nhead=1,
                dim_feedforward=f0_proj_size*4,
                dropout=f0_proj_dropout,
                batch_first=True
            )

        self.use_bi_lstm_cross_attn = use_bi_lstm_cross_attn
        if use_bi_lstm_cross_attn:
            print("Using Bi-LSTM Cross Attention!")
            self.audio_bi_lstm = nn.GRU(
                input_size=audio_proj_dim,
                hidden_size=audio_proj_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

            self.text_bi_lstm = nn.GRU(
                input_size=text_proj_dim,
                hidden_size=text_proj_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

            self.audio_attn = nn.MultiheadAttention(
                embed_dim=audio_proj_dim*2,
                num_heads=8,
                dropout=audio_proj_dropout,
                batch_first=True
            )

            self.audio_simple_att_proj = nn.Linear(audio_proj_dim*2, audio_proj_dim*2)

            self.text_attn = nn.MultiheadAttention(
                embed_dim=text_proj_dim*2,
                num_heads=8,
                dropout=text_proj_dropout,
                batch_first=True,
            )

            self.text_simple_att_proj = nn.Linear(text_proj_dim*2, text_proj_dim*2)

            if use_cross_f0_attn:
                print("Using Bi-LSTM Cross Attention for F0!")
                self.f0_bi_lstm = nn.GRU(
                    input_size=f0_proj_size,
                    hidden_size=f0_proj_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )

                self.f0_attn = nn.MultiheadAttention(
                    embed_dim=f0_proj_size*2,
                    num_heads=8,
                    dropout=f0_proj_dropout,
                    batch_first=True
                )

                self.f0_simple_att_proj = nn.Linear(f0_proj_size*2, f0_proj_size*2)

        self.use_hcam = use_hcam
        if use_hcam:
            print("Using Hierarchical Cross Attention Model (HCAM)!")
            self.audio_context_gru = nn.GRU(
                input_size=audio_proj_dim,
                hidden_size=audio_proj_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.text_context_gru = nn.GRU(
                input_size=text_proj_dim,
                hidden_size=text_proj_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            # Self‐attention after contextual GRU
            self.audio_self_attn = nn.MultiheadAttention(
                embed_dim=audio_proj_dim*2,  # because bidirectional
                num_heads=8,
                batch_first=True
            )
            self.text_self_attn = nn.MultiheadAttention(
                embed_dim=text_proj_dim*2,
                num_heads=8,
                batch_first=True
            )
            self.audio_fc_after_self = nn.Linear(audio_proj_dim*2, audio_proj_dim*2)
            self.audio_ln_after_self = nn.LayerNorm(audio_proj_dim*2)

            self.text_fc_after_self = nn.Linear(text_proj_dim*2, text_proj_dim*2)
            self.text_ln_after_self = nn.LayerNorm(text_proj_dim*2)

            # Cross‐attention blocks (audio→text and text→audio)
            self.cross_attn_a2t = nn.MultiheadAttention(
                embed_dim=audio_proj_dim*2,
                num_heads=8,
                batch_first=True
            )
            self.cross_attn_t2a = nn.MultiheadAttention(
                embed_dim=text_proj_dim*2,
                num_heads=8,
                batch_first=True
            )
            # Post cross‐attn transformations
            self.audio_fc_after_cross = nn.Linear(audio_proj_dim*2, audio_proj_dim*2)
            self.text_fc_after_cross = nn.Linear(text_proj_dim*2, text_proj_dim*2)
            self.audio_ln_after_cross = nn.LayerNorm(audio_proj_dim*2)
            self.text_ln_after_cross = nn.LayerNorm(text_proj_dim*2)

            # Projection layers for final pooling
            self.audio_attentive_pooling_proj = nn.Linear(audio_proj_dim*2, audio_proj_dim*2)
            self.text_attentive_pooling_proj = nn.Linear(text_proj_dim*2, text_proj_dim*2)

            if use_cross_f0_attn:
                print("Using Hierarchical Cross Attention Model (HCAM) for F0!")
                self.f0_context_gru = nn.GRU(
                    input_size=f0_proj_size,
                    hidden_size=f0_proj_size,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True
                )

                self.f0_self_attn = nn.MultiheadAttention(
                    embed_dim=f0_proj_size*2,
                    num_heads=8,
                    batch_first=True
                )

                self.f0_fc_after_self = nn.Linear(f0_proj_size*2, f0_proj_size*2)
                self.f0_ln_after_self = nn.LayerNorm(f0_proj_size*2)

                self.f0_attn = nn.MultiheadAttention(
                    embed_dim=f0_proj_size*2,
                    num_heads=8,
                    batch_first=True
                )

                self.f0_fc_after_cross = nn.Linear(f0_proj_size*2, f0_proj_size*2)
                self.f0_ln_after_cross = nn.LayerNorm(f0_proj_size*2)

                self.f0_attentive_pooling_proj = nn.Linear(f0_proj_size*2, f0_proj_size*2)

        self.f0_pooling_strategy = f0_pooling_strategy
        if f0_pooling_strategy == "attpool":
            # AttentiveStatisticsPooling requires initialization once we know F
            # Which is half of the MLP input dimension because we concatenate mean and std
            # Consider that mlp_input_dim is always equal to the F dimension of the embeddings
            self.f0_attpool = AttentiveStatisticsPooling(input_size=int(mlp_input_dim/2))

        self.use_mdat = use_mdat
        if use_mdat:
            print("Using MDAT (Graph-Att + Co-Att + optional Tx-Encoder).")
            self.audio_graph_attn = GraphAttentionLayer(
                in_features=audio_proj_dim,
                out_features=audio_proj_dim,
            )
            self.text_graph_attn = GraphAttentionLayer(
                in_features=text_proj_dim,
                out_features=text_proj_dim,
            )
            # Co-Attention
            self.co_attention = CoAttentionLayer(d_model=audio_proj_dim)
            # Optional final Transformer Encoders (as in the paper)
            self.audio_trans_enc = nn.TransformerEncoderLayer(
                d_model=audio_proj_dim,
                nhead=8,
                dim_feedforward=audio_proj_dim*4,
                dropout=audio_proj_dropout,
                batch_first=True
            )
            self.text_trans_enc = nn.TransformerEncoderLayer(
                d_model=text_proj_dim,
                nhead=8,
                dim_feedforward=text_proj_dim*4,
                dropout=text_proj_dropout,
                batch_first=True
            )

            # Projection layers for final pooling
            self.audio_attentive_pooling_proj = nn.Linear(audio_proj_dim, audio_proj_dim)
            self.text_attentive_pooling_proj = nn.Linear(text_proj_dim, text_proj_dim)

            self.use_mdat_multihead = use_mdat_multihead
            if use_mdat_multihead:
                print("Using MDAT with Multi-Head Cross-Attention!")
                # We can still reuse graph-attention or define a multi-head version
                self.audio_graph_attn = GraphAttentionLayer(
                    in_features=audio_proj_dim,
                    out_features=audio_proj_dim
                )
                self.text_graph_attn = GraphAttentionLayer(
                    in_features=text_proj_dim,
                    out_features=text_proj_dim
                )
                # Multihead cross-attention layers
                self.mda_audio_to_text = nn.MultiheadAttention(
                    embed_dim=audio_proj_dim,
                    num_heads=8,               # (Adjust as desired)
                    dropout=audio_proj_dropout,
                    batch_first=True
                )
                self.mda_text_to_audio = nn.MultiheadAttention(
                    embed_dim=text_proj_dim,
                    num_heads=8,               # (Adjust as desired)
                    dropout=text_proj_dropout,
                    batch_first=True
                )

                self.use_mdat_multihead_gru = use_mdat_multihead_gru
                if use_mdat_multihead_gru:
                    print("Using MDAT with Multi-Head Cross-Attention and GRU!")
                    self.audio_gru = nn.GRU(
                        input_size=audio_proj_dim,
                        hidden_size=audio_proj_dim,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=True
                    )
                    self.text_gru = nn.GRU(
                        input_size=text_proj_dim,
                        hidden_size=text_proj_dim,
                        num_layers=1,
                        batch_first=True,
                        bidirectional=True
                    )

                    # Projection layers for final pooling
                    self.audio_attentive_pooling_proj_mha = nn.Linear(audio_proj_dim*2, audio_proj_dim*2)
                    self.text_attentive_pooling_proj_mha  = nn.Linear(text_proj_dim*2,  text_proj_dim*2)
                else:
                    # Optional TransformerEncoder layers (as in MDAT):
                    self.audio_trans_enc_mha = nn.TransformerEncoderLayer(
                        d_model=audio_proj_dim,
                        nhead=8,   # also multi-head
                        dim_feedforward=4 * audio_proj_dim,
                        dropout=audio_proj_dropout,
                        batch_first=True
                    )
                    self.text_trans_enc_mha = nn.TransformerEncoderLayer(
                        d_model=text_proj_dim,
                        nhead=8,   # also multi-head
                        dim_feedforward=4 * text_proj_dim,
                        dropout=text_proj_dropout,
                        batch_first=True
                    )

                    # Final learnable projection for attentive pooling
                    self.audio_attentive_pooling_proj_mha = nn.Linear(audio_proj_dim, audio_proj_dim)
                    self.text_attentive_pooling_proj_mha  = nn.Linear(text_proj_dim,  text_proj_dim)

    def _get_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        wavs_flag = False
        audio, text, f0 = x
        if isinstance(audio, dict) and "whisper_feature" in audio:
            whisper_feature = audio["whisper_feature"]
            whisper_compression = audio["whisper_compression"]
            audio = whisper_feature
        # Remove "wavs" in case the model is Whisper
        if "wavs" in audio:
            wavs = audio.pop("wavs")
            wavs_flag = True
        if self.freeze_audio_backbone:
            with torch.no_grad():
                audio_hidden_states = self.audio_backbone(**audio).last_hidden_state
        else:
            audio_hidden_states = self.audio_backbone(**audio).last_hidden_state

        if "whisper" in self.audio_model_name.lower():
            audio_hidden_states = audio_hidden_states[:, : whisper_compression, :]

        if self.freeze_text_backbone:
            with torch.no_grad():
                text_hidden_states = self.text_backbone(**text).last_hidden_state
        else:
            text_hidden_states = self.text_backbone(**text).last_hidden_state

        if wavs_flag:
            audio["wavs"] = wavs

        return audio_hidden_states, text_hidden_states, f0

    def _get_mel_spec_embeddings(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio, _, _ = x
        if "whisper_feature" in audio:
            audio = audio["whisper_feature"]
        if "wavs" in audio:
            audio = audio["wavs"]
        # "input_features" is specific for whisper
        elif "input_features" in audio:
            audio = audio["input_features"]
        else:
            audio = audio["input_values"]
        mel_spec_embeddings = self.mel_spec_encoder(audio)
        return mel_spec_embeddings

    def apply_transformer_enc(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor, f0_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Projection
        audio_hidden_states = self.audio_proj(audio_hidden_states)
        text_hidden_states = self.text_proj(text_hidden_states)
        f0_hidden_states = self.f0_proj(f0_hidden_states)

        # Transformer Encoder
        audio_hidden_states = self.audio_trans_enc(audio_hidden_states)
        text_hidden_states = self.text_trans_enc(text_hidden_states)
        f0_hidden_states = self.f0_trans_enc(f0_hidden_states)

        # Mean pooling over T
        audio_embeddings = audio_hidden_states.mean(dim=1)
        text_embeddings = text_hidden_states.mean(dim=1)
        f0_embeddings = f0_hidden_states.mean(dim=1)

        return audio_embeddings, text_embeddings, f0_embeddings

    def apply_bi_lstm_cross_attn(self, audio_hidden_states: torch.Tensor, text_hidden_states: torch.Tensor, f0_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Projection
        audio_hidden_states = self.audio_proj(audio_hidden_states)
        text_hidden_states = self.text_proj(text_hidden_states)
        f0_hidden_states = self.f0_proj(f0_hidden_states)

        # Bi-LSTM
        audio_hidden_states, _ = self.audio_bi_lstm(audio_hidden_states)
        text_hidden_states, _ = self.text_bi_lstm(text_hidden_states)

        # Multihead Attention
        att_audio_hidden_states, _ = self.audio_attn(audio_hidden_states, text_hidden_states, text_hidden_states)
        att_text_hidden_states, _ = self.text_attn(text_hidden_states, audio_hidden_states, audio_hidden_states)

        audio_hidden_states = audio_hidden_states + att_audio_hidden_states
        text_hidden_states = text_hidden_states + att_text_hidden_states

        if self.use_cross_f0_attn:
            # Bi-LSTM
            f0_hidden_states, _ = self.f0_bi_lstm(f0_hidden_states)

            # Multihead Attention
            att_f0_hidden_states_audio, _ = self.f0_attn(f0_hidden_states, audio_hidden_states, audio_hidden_states)
            att_f0_hidden_states_text, _ = self.f0_attn(f0_hidden_states, text_hidden_states, text_hidden_states)

            # Attention pooling
            f0_hidden_states = f0_hidden_states + att_f0_hidden_states_audio + att_f0_hidden_states_text
            f0_att = self.f0_simple_att_proj(f0_hidden_states)
            f0_weights = F.softmax(f0_att, dim=1)
            f0_embeddings = torch.sum(f0_weights * f0_hidden_states, dim=1)
        else:
            f0_embeddings = f0_hidden_states.mean(dim=1)

        audio_att = self.audio_simple_att_proj(audio_hidden_states)
        text_att = self.text_simple_att_proj(text_hidden_states)

        audio_weights = F.softmax(audio_att, dim=1)
        text_weights = F.softmax(text_att, dim=1)

        audio_embeddings = torch.sum(audio_weights * audio_hidden_states, dim=1)
        text_embeddings = torch.sum(text_weights * text_hidden_states, dim=1)

        return audio_embeddings, text_embeddings, f0_embeddings

    def apply_hcam(self, audio_hidden_states, text_hidden_states, f0_hidden_states):
        """
        Implements a simplified “HCAM” pipeline:
          1) Project each modality.
          2) Contextual Bi‐GRU for each (Stage II style).
          3) Self‐attention on each modality.
          4) Cross‐attention between modalities.
          5) Final pooling + optional concatenation into a single embedding.
        """
        # Projection
        A = self.audio_proj(audio_hidden_states)  # [B,T,a_proj]
        T = self.text_proj(text_hidden_states)    # [B,T,t_proj]
        F0 = self.f0_proj(f0_hidden_states)       # [B,T,f0_proj]

        # GRU for contextual information
        A, _ = self.audio_context_gru(A)  # [B,T,2*a_proj]
        T, _ = self.text_context_gru(T)   # [B,T,2*t_proj]

        # Audio Self‐attention
        A_sa, _ = self.audio_self_attn(A, A, A)
        A = A + A_sa
        A_fc = self.audio_fc_after_self(A)
        # Skip connection
        A = A + A_fc
        A = self.audio_ln_after_self(A)

        # Text Self‐attention
        T_sa, _ = self.text_self_attn(T, T, T)
        T = T + T_sa
        T_fc = self.text_fc_after_self(T)
        # Skip connection
        T = T + T_fc
        T = self.text_ln_after_self(T)

        # Audio Cross‐attention
        A_ca, _ = self.cross_attn_a2t(A, T, T)  # cross from text
        A = A + A_ca
        A_fc2 = self.audio_fc_after_cross(A)
        A = A + A_fc2
        A = self.audio_ln_after_cross(A)

        # Text Cross‐attention
        T_ca, _ = self.cross_attn_t2a(T, A, A)
        T = T + T_ca
        T_fc2 = self.text_fc_after_cross(T)
        T = T + T_fc2
        T = self.text_ln_after_cross(T)

        # Attention pooling
        A_weigths = F.softmax(self.audio_attentive_pooling_proj(A), dim=1)
        T_weigths = F.softmax(self.text_attentive_pooling_proj(T), dim=1)

        audio_embeddings = torch.sum(A_weigths * A, dim=1)
        text_embeddings = torch.sum(T_weigths * T, dim=1)

        if self.use_cross_f0_attn:
            # GRU for contextual information
            F0, _ = self.f0_context_gru(F0)  # [B,T,2*f0_proj]

            # F0 Self‐attention
            F0_sa, _ = self.f0_self_attn(F0, F0, F0)
            F0 = F0 + F0_sa
            F0_fc = self.f0_fc_after_self(F0)
            # Skip connection
            F0 = F0 + F0_fc
            F0 = self.f0_ln_after_self(F0)

            # F0-Audio Cross‐attention
            F0_ca_A, _ = self.f0_attn(F0, A, A)
            F0_A = F0 + F0_ca_A
            F0_fc_A = self.f0_fc_after_cross(F0_A)
            F0_A = F0_A + F0_fc_A
            F0_A = self.f0_ln_after_cross(F0_A)

            # F0-Text Cross‐attention
            F0_ca_T, _ = self.f0_attn(F0, T, T)
            F0_T = F0 + F0_ca_T
            F0_fc_T = self.f0_fc_after_cross(F0_T)
            F0_T = F0_T + F0_fc_T
            F0_T = self.f0_ln_after_cross(F0_T)

            # Attention pooling
            F0 = F0 + F0_A + F0_T

            # Attention pooling
            F0_weigths = F.softmax(self.f0_attentive_pooling_proj(F0), dim=1)
            f0_embeddings = torch.sum(F0_weigths * F0, dim=1)
        else:
            f0_embeddings = F0.mean(dim=1)

        return audio_embeddings, text_embeddings, f0_embeddings

    def apply_mdat(
        self,
        audio_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
        f0_hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Implements the MDAT pipeline:
          1) Project audio/text
          2) Graph Attention for each modality
          3) Co-Attention across modalities
          4) Transformer Encoder layers (per modality)
          5) Pool the final sequences (e.g. mean)
        """
        # Projection
        A = self.audio_proj(audio_hidden_states)  # [B, Ta, Da]
        T = self.text_proj(text_hidden_states)    # [B, Tt, Dt]
        F0 = self.f0_proj(f0_hidden_states)       # [B, Tf, Df]

        # Graph attention for each modality
        A_g = self.audio_graph_attn(A)  # [B, Ta, Da]
        T_g = self.text_graph_attn(T)   # [B, Tt, Dt]

        # Co-attention
        coA, coT = self.co_attention(A_g, T_g)  # [B, Ta, Da], [B, Tt, Dt]

        # Skip connections
        A_co = A_g + coA
        T_co = T_g + coT

        # Transformer encoders
        A_enc = self.audio_trans_enc(A_co)  # [B, Ta, Da]
        T_enc = self.text_trans_enc(T_co)   # [B, Tt, Dt]

        # Attentive pooling
        A_weigths = F.softmax(self.audio_attentive_pooling_proj(A_enc), dim=1)
        T_weigths = F.softmax(self.text_attentive_pooling_proj(T_enc), dim=1)

        audio_embeddings = torch.sum(A_weigths * A_enc, dim=1)
        text_embeddings = torch.sum(T_weigths * T_enc, dim=1)

        if self.use_cross_f0_attn:
            # Graph attention for f0
            F0_g = self.f0_graph_attn(F0)

            # Co-Attention for f0 and audio
            coF0, coA = self.f0_co_attention(F0_g, A_g)
            F0_co_A = F0_g + coF0

            # Co-Attention for f0 and text
            coF0, coT = self.f0_co_attention(F0_g, T_g)
            F0_co_T = F0_g + coF0

            # Skip connections
            F0_co = F0_g + F0_co_A + F0_co_T

            # Transformer encoders
            F0_enc = self.f0_trans_enc(F0_co)  # [B, Tf, Df]

            # Attentive pooling
            F0_weigths = F.softmax(self.f0_attentive_pooling_proj(F0_enc), dim=1)

            f0_embeddings = torch.sum(F0_weigths * F0_enc, dim=1)
        else:
            f0_embeddings = F0.mean(dim=1)

        return audio_embeddings, text_embeddings, f0_embeddings

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        audio_hidden_states, text_hidden_states, f0 = self._get_embeddings(x)
        f0_embeddings = self.f0_embedding(f0)
        # check if model is using transformers base pooling
        if self.use_transformer_enc:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_transformer_enc(audio_hidden_states, text_hidden_states, f0_embeddings)
        elif self.use_bi_lstm_cross_attn:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_bi_lstm_cross_attn(audio_hidden_states, text_hidden_states, f0_embeddings)
        elif self.use_hcam:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_hcam(audio_hidden_states, text_hidden_states, f0_embeddings)
        elif self.use_mdat:
            audio_embeddings, text_embeddings, f0_embeddings = self.apply_mdat(audio_hidden_states, text_hidden_states, f0_embeddings)
        else:
            # Apply layer weighting
            audio_embeddings, text_embeddings, f0_embeddings = self._apply_pooling(audio_hidden_states, text_hidden_states, f0_embeddings)
            # Audio projection
            audio_embeddings = self.audio_proj(audio_embeddings)
            # Text projection
            text_embeddings = self.text_proj(text_embeddings)
            # F0 projection
            f0_embeddings = self.f0_proj(f0_embeddings)
        # Mel Spec embeddings
        mel_spec_embeddings = self._get_mel_spec_embeddings(x)
        # Concatenate audio and text embeddings
        logits_input = torch.cat((audio_embeddings, text_embeddings, f0_embeddings, mel_spec_embeddings), dim=-1) # [B,AF+TF]
        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)
        return logits