import os
import random
from typing import List, Tuple, Dict, Optional

import torch
import torchaudio
import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw
from torch.utils.data import Dataset
from transformers import WhisperFeatureExtractor
from torch_audiomentations import AddBackgroundNoise, ApplyImpulseResponse, Identity


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        filename_column: str,
        target_column: str,
        base_dir: str,
        use_seqaug: bool = False,
        data_type: str = "train",
    ):
        """Initialization"""
        self.data = data

        # Cache filepaths and targets
        self.filenames = self.data[filename_column].values
        self.targets = self.data[target_column].values

        self.filename_column = filename_column
        self.target_column = target_column

        self.use_seqaug = use_seqaug

        self.base_dir = base_dir
        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def _load_file(self, filepath: str) -> torch.Tensor:
        """Load an audio file

        Params:

        filepath (str): Path to the audio file

        Returns:

        torch.Tensor: Audio tensor
        """
        features = torch.load(filepath)

        return features

    def seqaug(self, input_tensor, alpha: float = 0.2):
        """
        Applies SeqAug (https://arxiv.org/abs/2305.01954) augmentation to the input tensor.

        Params:
            input_tensor (torch.Tensor): Input tensor of shape [sequence_length, feature_size].
            alpha (float): Parameter for the Beta distribution (α ∈ [0, 1]).

        Returns:
            output_tensor (torch.Tensor): Augmented tensor of the same shape as input_tensor.
        """
        # Get dimensions
        sequence_length, feature_size = input_tensor.shape

        # Sample proportion p from Beta(α, α)
        beta_dist = torch.distributions.Beta(alpha, alpha)
        p = beta_dist.sample()

        # Determine the number of feature addresses to sample
        num_features_to_sample = int(p.item() * feature_size)
        num_features_to_sample = max(num_features_to_sample, 1)  # Ensure at least one feature is selected

        # Randomly select feature addresses (indices) to permute
        selected_features = torch.randperm(feature_size)[:num_features_to_sample]

        # Generate a random permutation of the time indices
        perm = torch.randperm(sequence_length)

        # Create a copy of the input tensor to hold the augmented data
        output_tensor = input_tensor.clone()

        # Permute the selected features along the time axis according to the random permutation
        output_tensor[:, selected_features] = input_tensor[perm][:, selected_features]

        return output_tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[str]]:
        """Get an item from the dataset

        Params:

        index (int): Index of the item to get

        Returns:

        Dict[torch.Tensor, np.ndarray]: A dictionary containing the audio and the caption
        """
        filename = self.filenames[index]
        filepath = os.path.join(self.base_dir, filename)

        target = self.targets[index]

        features = self._load_file(filepath)

        if self.use_seqaug and self.data_type == "train":
            features = self.seqaug(features)

        # Convert features to float
        features = features.float()

        return features, target


class EmbeddingCollate:
    def __init__(
        self,
        padding_value: float = 0.0,
    ):
        """
        Collation function for dynamic batching of audio data.

        Params:
            padding_value (float): Value to use for padding shorter sequences.
        """
        self.padding_value = padding_value

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features, targets = zip(*batch)
        batch_size = len(features)
        num_layers = features[0].shape[0]
        feature_dim = features[0].shape[-1]

        features = list(features)
        targets = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in targets])

        lengths = [feature.shape[1] for feature in features]
        max_length = max(lengths)

        padded_features = torch.full((batch_size, num_layers, max_length, feature_dim), self.padding_value)

        for i, feature in enumerate(features):
            length = feature.shape[1]
            padded_features[i, :, :length, :] = feature

        return padded_features, targets


class LastLayerEmbeddingDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        filename_column: str,
        target_column: str,
        base_dir: str,
    ):
        """Initialization"""
        self.data = data

        # Cache filepaths and targets
        self.filenames = self.data[filename_column].values
        self.targets = self.data[target_column].values

        self.filename_column = filename_column
        self.target_column = target_column

        self.base_dir = base_dir

    def __len__(self):
        return len(self.data)

    def _load_file(self, filepath: str) -> torch.Tensor:
        """Load an audio file

        Params:

        filepath (str): Path to the audio file

        Returns:

        torch.Tensor: Audio tensor
        """
        features = torch.load(filepath)
        return features

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[str]]:
        """Get an item from the dataset

        Params:

        index (int): Index of the item to get

        Returns:

        Dict[torch.Tensor, np.ndarray]: A dictionary containing the audio and the caption
        """
        filename = self.filenames[index]

        if filename.endswith(".wav"):
            filename = filename[:-4] + ".pt"

        filepath = os.path.join(self.base_dir, filename)

        target = self.targets[index]

        features = self._load_file(filepath)

        # Convert features to float
        features = features.float()

        return features, target


class LastLayerEmbeddingCollate:
    def __init__(self, padding_value: float = 0.0):
        self.padding_value = padding_value

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        features, targets = zip(*batch)
        targets = torch.stack([torch.as_tensor(t) for t in targets])

        processed_features = []
        for feature in features:
            # Only check and fix shape if it's (1, T, F)
            if feature.dim() == 3 and feature.shape[0] == 1:
                feature = feature.squeeze(0)  # (T, F)
            processed_features.append(feature)

        features = processed_features
        lengths = [f.shape[0] for f in features]
        max_length = max(lengths)

        feature_dim = features[0].shape[-1]

        padded_features = torch.full((len(features), max_length, feature_dim), self.padding_value)
        for i, f in enumerate(features):
            padded_features[i, :f.shape[0]] = f

        return padded_features, targets


class DynamicDataset(Dataset):
    def __init__(
        self,
        data,
        base_dir: str,
        filename_column: str,
        target_column: str,
        class_num: int = 15,
        target_sr: int = 16000,
        data_type: str = "train",
        # Mixup parameters (Optional)
        mixup_alpha: Optional[float] = 0.0,
        # Random truncation parameters (Optional)
        use_rand_truncation: Optional[bool] = False,
        min_duration: Optional[float] = 0.0,
        # background noise parameters (Optional)
        use_background_noise: Optional[bool] = False,
        background_noise_dir: Optional[str] = None,
        background_noise_min_snr_in_db: Optional[float] = 3.0,
        background_noise_max_snr_in_db: Optional[float] = 15.0,
        background_noise_p: Optional[float] = 0.5,
        # impulse response parameters (Optional)
        use_rir: Optional[bool] = False,
        rir_dir: Optional[str] = None,
        rir_p: Optional[float] = 0.5,
    ):
        """Initialization"""
        self.data = data
        self.base_dir = base_dir

        # Cache filepaths and targets
        self.filenames = self.data[filename_column].values
        self.targets = self.data[target_column].values

        self.filename_column = filename_column
        self.target_column = target_column

        # Data augmentation parameters
        self.mixup_alpha = mixup_alpha

        self.min_duration = min_duration
        self.use_rand_truncation = use_rand_truncation

        self.data_type = data_type
        self.class_num = class_num

        self.target_sr = target_sr
        # Cache for sampling rate resamplers
        self.resamplers = {}

        # Apply background noise
        if use_background_noise:
            self.background_noise = AddBackgroundNoise(
                background_paths=background_noise_dir,
                min_snr_in_db=background_noise_min_snr_in_db,
                max_snr_in_db=background_noise_max_snr_in_db,
                sample_rate=target_sr,
                target_rate=target_sr,
                p=background_noise_p
            )
        else:
            # Identity augmentation if not using background noise
            self.background_noise = Identity()

        # Apply impulse response
        if use_rir:
            self.impulse_response = ApplyImpulseResponse(
                ir_paths=rir_dir,
                sample_rate=target_sr,
                target_rate=target_sr,
                p=rir_p,
            )
        else:
            # Identity augmentation if not using impulse response
            self.impulse_response = Identity()

    def __len__(self):
        return len(self.data)

    def _random_truncation(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Cut or pad an audio tensor to the desired length.
        """
        min_length = int(self.min_duration * self.target_sr)
        len_audio = audio.shape[-1]

        if self.use_rand_truncation and len_audio > min_length and random.random() < 0.5:
            segment_length = random.randint(min_length, len_audio)

            max_start = len_audio - segment_length
            start = random.randint(0, max_start)
            end = start + segment_length

            audio = audio[..., start:end]

        return audio

    def _load_wav(self, filepath: str):
        waveform, source_sr = torchaudio.load(filepath)

        # Convert to mono if stereo
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if source_sr != self.target_sr:
            if source_sr not in self.resamplers:
                self.resamplers[source_sr] = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=self.target_sr)
            waveform = self.resamplers[source_sr](waveform)

        return waveform, self.target_sr

    def __getitem__(self, index: int) -> Dict[torch.Tensor, torch.Tensor]:
        main_target = self.targets[index]
        main_file = self.filenames[index]

        # If using mixup and in training mode
        if self.mixup_alpha > 0.0 and self.data_type == "train" and random.random() < self.mixup_alpha:
            attempts = 0
            rand_index = index
            # Force mixup with a different targets, attempt limit is 10 to avoid infinite loop
            while attempts < 10 and self.targets[rand_index] == main_target:
                rand_index = random.randint(0, len(self.targets) - 1)
                attempts += 1

            rand_target = self.targets[rand_index]
            rand_file = self.filenames[rand_index]

            original_path = os.path.join(self.base_dir, main_file)
            rand_path = os.path.join(self.base_dir, rand_file)

            audio_original, _ = self._load_wav(original_path)
            audio_rand, _ = self._load_wav(rand_path)

            # Sample lambda from beta distribution
            mix_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            # Audios must have the same length
            if audio_original.shape[-1] > audio_rand.shape[-1]:
                audio_rand = torch.nn.functional.pad(audio_rand, (0, audio_original.shape[-1] - audio_rand.shape[-1]))
            elif audio_original.shape[-1] < audio_rand.shape[-1]:
                audio_original = torch.nn.functional.pad(audio_original, (0, audio_rand.shape[-1] - audio_original.shape[-1]))

            # Mixup
            audio = mix_lambda * audio_original + (1 - mix_lambda) * audio_rand

            # When using mixup we need to use one-hot encoding for the target
            target = torch.zeros(self.class_num)
            target[main_target] = mix_lambda
            target[rand_target] = 1 - mix_lambda
        else:
            filepath = os.path.join(self.base_dir, main_file)
            audio, _ = self._load_wav(filepath)
            target = main_target

        # One-hot encoding for the target if using mixup in eval mode
        if self.mixup_alpha > 0.0 and self.data_type != "train":
            target = torch.zeros(self.class_num)
            target[main_target] = 1.0
        # Random Truncation
        if self.use_rand_truncation and self.data_type == "train":
            audio = self._random_truncation(audio)
        # Background noise insertion or Identity
        audio = self.background_noise(audio.unsqueeze(0)).squeeze(0)
        # Impulse response (Reverberation) or Identity
        audio = self.impulse_response(audio.unsqueeze(0)).squeeze(0)
        return audio.squeeze(0).numpy(), target


class DynamicCollate:
    def __init__(
        self,
        padding_value: float = 0.0,
        processor = None,
        target_sr: int = 16000
    ):
        """
        Collation function for dynamic batching of audio data.

        Params:
            padding_value (float): Value to use for padding shorter sequences.
            processor: A processor or feature extractor to process raw audio
                       into features if desired.
        """
        self.processor = processor
        self.target_sr = target_sr
        self.padding_value = padding_value

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        audios, targets = zip(*batch)

        audios = list(audios)
        targets = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in targets])

        processed = self.processor(
            audios,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        )

        # Special case for Whisper, that expects a fixed input size of 3000 (30 seconds)
        if isinstance(self.processor, WhisperFeatureExtractor) and processed.input_features.shape[-1] < 3000:
            processed = self.processor(
                audios,
                return_tensors="pt",
                sampling_rate=self.target_sr,
            )

        return processed, targets


class DynamicAudioTextDataset(DynamicDataset):
    def __init__(
        self,
        transcript_column: str,
        # text augmentation parameters (Optional)
        use_text_augmentation: Optional[bool] = False,
        text_augmentation_p: Optional[float] = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        """Initialization"""
        self.transcripts = self.data[transcript_column].values
        self.use_text_augmentation = use_text_augmentation
        # Text augmentation
        if self.use_text_augmentation:
            self.text_augmenter = naw.RandomWordAug()
            self.text_augmentation_p = text_augmentation_p

    def __getitem__(self, index: int) -> Dict[torch.Tensor, torch.Tensor]:
        main_target = self.targets[index]
        main_file = self.filenames[index]
        transcript = self.transcripts[index]

        # Apply text augmentation
        if self.use_text_augmentation and self.data_type == "train" and random.random() < self.text_augmentation_p:
            transcript = self.text_augmenter.augment(transcript)[0]

        # If using mixup and in training mode
        if self.mixup_alpha > 0.0 and self.data_type == "train" and random.random() < self.mixup_alpha:
            attempts = 0
            rand_index = index
            # Force mixup with a different targets, attempt limit is 10 to avoid infinite loop
            while attempts < 10 and self.targets[rand_index] == main_target:
                rand_index = random.randint(0, len(self.targets) - 1)
                attempts += 1

            rand_target = self.targets[rand_index]
            rand_file = self.filenames[rand_index]

            original_path = os.path.join(self.base_dir, main_file)
            rand_path = os.path.join(self.base_dir, rand_file)

            audio_original, _ = self._load_wav(original_path)
            audio_rand, _ = self._load_wav(rand_path)

            # Sample lambda from beta distribution
            mix_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            # Audios must have the same length
            if audio_original.shape[-1] > audio_rand.shape[-1]:
                audio_rand = torch.nn.functional.pad(audio_rand, (0, audio_original.shape[-1] - audio_rand.shape[-1]))
            elif audio_original.shape[-1] < audio_rand.shape[-1]:
                audio_original = torch.nn.functional.pad(audio_original, (0, audio_rand.shape[-1] - audio_original.shape[-1]))

            # Mixup
            audio = mix_lambda * audio_original + (1 - mix_lambda) * audio_rand

            # When using mixup we need to use one-hot encoding for the target
            target = torch.zeros(self.class_num)
            target[main_target] = mix_lambda
            target[rand_target] = 1 - mix_lambda
        else:
            filepath = os.path.join(self.base_dir, main_file)
            audio, _ = self._load_wav(filepath)
            target = main_target

        # One-hot encoding for the target if using mixup in eval mode (because we are using BCEWithLogitsLoss)
        if self.mixup_alpha > 0.0 and self.data_type != "train":
            target = torch.zeros(self.class_num)
            target[main_target] = 1.0

        # Random Truncation
        if self.use_rand_truncation and self.data_type == "train":
            audio = self._random_truncation(audio)
        # Background noise insertion or Identity
        audio = self.background_noise(audio.unsqueeze(0)).squeeze(0)
        # Impulse response (Reverberation) or Identity
        audio = self.impulse_response(audio.unsqueeze(0)).squeeze(0)

        return audio.squeeze(0).numpy(), transcript, target


class DynamicAudioTextCollate:
    def __init__(
        self,
        padding_value: float = 0.0,
        processor = None,
        text_tokenizer = None,
        target_sr: int = 16000
    ):
        """
        Collation function for dynamic batching of audio data.

        Params:
            padding_value (float): Value to use for padding shorter sequences.
            processor: A processor or feature extractor to process raw audio
                       into features if desired.
        """
        self.processor = processor
        self.text_tokenizer = text_tokenizer
        self.target_sr = target_sr
        self.padding_value = padding_value

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        audios, transcripts, targets = zip(*batch)

        audios = list(audios)
        targets = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in targets])

        processed = self.processor(
            audios,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        )

        tokenized_transcripts = self.text_tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Special case for Whisper, that expects a fixed input size of 3000 (30 seconds)
        if isinstance(self.processor, WhisperFeatureExtractor) and processed.input_features.shape[-1] < 3000:
            processed = self.processor(
                audios,
                return_tensors="pt",
                sampling_rate=self.target_sr,
            )

        return (processed, tokenized_transcripts), targets


class DynamicAudioTextSpeakerEmbDataset(DynamicDataset):
    def __init__(
        self,
        transcript_column: str,
        speakeremb_base_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        """Initialization"""
        self.speakeremb_base_dir = speakeremb_base_dir
        self.transcripts = self.data[transcript_column].values

    def __getitem__(self, index: int) -> Dict[torch.Tensor, torch.Tensor]:
        main_target = self.targets[index]
        main_file = self.filenames[index]
        transcript = self.transcripts[index]

        speaker_emb = torch.load(os.path.join(self.speakeremb_base_dir, main_file.replace(".wav", ".pt")))
        if speaker_emb.dim() == 2:
            speaker_emb = speaker_emb.squeeze(0)

        # If using mixup and in training mode
        if self.mixup_alpha > 0.0 and self.data_type == "train" and random.random() < self.mixup_alpha:
            attempts = 0
            rand_index = index
            # Force mixup with a different targets, attempt limit is 10 to avoid infinite loop
            while attempts < 10 and self.targets[rand_index] == main_target:
                rand_index = random.randint(0, len(self.targets) - 1)
                attempts += 1

            rand_target = self.targets[rand_index]
            rand_file = self.filenames[rand_index]

            original_path = os.path.join(self.base_dir, main_file)
            rand_path = os.path.join(self.base_dir, rand_file)

            audio_original, _ = self._load_wav(original_path)
            audio_rand, _ = self._load_wav(rand_path)

            # Sample lambda from beta distribution
            mix_lambda = np.random.beta(self.mixup_alpha, self.mixup_alpha)

            # Audios must have the same length
            if audio_original.shape[-1] > audio_rand.shape[-1]:
                audio_rand = torch.nn.functional.pad(audio_rand, (0, audio_original.shape[-1] - audio_rand.shape[-1]))
            elif audio_original.shape[-1] < audio_rand.shape[-1]:
                audio_original = torch.nn.functional.pad(audio_original, (0, audio_rand.shape[-1] - audio_original.shape[-1]))

            # Mixup
            audio = mix_lambda * audio_original + (1 - mix_lambda) * audio_rand

            # When using mixup we need to use one-hot encoding for the target
            target = torch.zeros(self.class_num)
            target[main_target] = mix_lambda
            target[rand_target] = 1 - mix_lambda
        else:
            filepath = os.path.join(self.base_dir, main_file)
            audio, _ = self._load_wav(filepath)
            target = main_target

        # One-hot encoding for the target if using mixup in eval mode
        if self.mixup_alpha > 0.0 and self.data_type != "train":
            target = torch.zeros(self.class_num)
            target[main_target] = 1.0

        # Random Truncation
        if self.use_rand_truncation and self.data_type == "train":
            audio = self._random_truncation(audio)
        # Background noise insertion or Identity
        audio = self.background_noise(audio.unsqueeze(0)).squeeze(0)
        # Impulse response (Reverberation) or Identity
        audio = self.impulse_response(audio.unsqueeze(0)).squeeze(0)

        return audio.squeeze(0).numpy(), transcript, speaker_emb, target


class DynamicAudioTextSpeakerEmbCollate:
    def __init__(
        self,
        padding_value: float = 0.0,
        processor = None,
        text_tokenizer = None,
        target_sr: int = 16000
    ):
        """
        Collation function for dynamic batching of audio data.

        Params:
            padding_value (float): Value to use for padding shorter sequences.
            processor: A processor or feature extractor to process raw audio
                       into features if desired.
        """
        self.processor = processor
        self.text_tokenizer = text_tokenizer
        self.target_sr = target_sr
        self.padding_value = padding_value

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        audios, transcripts, speaker_embs, targets = zip(*batch)

        audios = list(audios)
        targets = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in targets])

        speaker_embs = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in speaker_embs])

        processed = self.processor(
            audios,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        )

        tokenized_transcripts = self.text_tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Special case for Whisper, that expects a fixed input size of 3000 (30 seconds)
        if isinstance(self.processor, WhisperFeatureExtractor) and processed.input_features.shape[-1] < 3000:
            processed = self.processor(
                audios,
                return_tensors="pt",
                sampling_rate=self.target_sr,
            )

        return (processed, tokenized_transcripts, speaker_embs), targets


class XEUSNestCollate:
    def __init__(
        self,
        padding_value: float = 0.0,
    ):
        """
        Collation function for dynamic batching of audio data.

        Params:
            padding_value (float): Value to use for padding shorter sequences.
            processor: A processor or feature extractor to process raw audio
                       into features if desired.
        """
        self.padding_value = padding_value

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        audios, targets = zip(*batch)

        audios = [torch.from_numpy(audio) if isinstance(audio, np.ndarray) else audio for audio in audios]
        targets = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in targets])
        # pad audios ana take the length
        audios_lengths = [audio.shape[-1] for audio in audios]
        # pad audios
        max_length = max(audios_lengths)

        padded_audios = torch.full((len(audios), max_length), self.padding_value)

        for i, audio in enumerate(audios):
            padded_audios[i, :audio.shape[-1]] = audio.float()

        processed = {
            "wavs": padded_audios,
            "wav_lengths": torch.tensor(audios_lengths),
        }

        return processed, targets


class LastLayerEmbeddingTextDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        filename_column: str,
        target_column: str,
        gender_column: str,
        transcript_column: str,
        base_dir: str,
        data_type: str = "val",
        # text augmentation parameters (Optional)
        use_text_augmentation: Optional[bool] = False,
        text_augmentation_p: Optional[float] = 0.5,
    ):
        """Initialization"""
        self.data = data

        # Cache filepaths and targets
        self.filenames = self.data[filename_column].values
        self.targets = self.data[target_column].values
        self.genders = self.data[gender_column].values

        self.filename_column = filename_column
        self.target_column = target_column

        self.base_dir = base_dir

        self.transcripts = self.data[transcript_column].values
        self.use_text_augmentation = use_text_augmentation
        # Text augmentation
        if self.use_text_augmentation:
            self.text_augmenter = naw.RandomWordAug()
            self.text_augmentation_p = text_augmentation_p

        self.data_type = data_type

    def __len__(self):
        return len(self.data)

    def _load_file(self, filepath: str) -> torch.Tensor:
        """Load an audio file

        Params:

        filepath (str): Path to the audio file

        Returns:

        torch.Tensor: Audio tensor
        """
        features = torch.load(filepath)
        return features

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[str]]:
        """Get an item from the dataset

        Params:

        index (int): Index of the item to get

        Returns:

        Dict[torch.Tensor, np.ndarray]: A dictionary containing the audio and the caption
        """
        filename = self.filenames[index]
        if filename.endswith(".wav"):
            filename = filename[:-4] + ".pt"
        filepath = os.path.join(self.base_dir, filename)

        target = self.targets[index]
        transcript = self.transcripts[index]

        # Apply text augmentation
        if self.use_text_augmentation and self.data_type == "train" and random.random() < self.text_augmentation_p:
            transcript = self.text_augmenter.augment(transcript)[0]

        features = self._load_file(filepath)

        # Convert features to float
        features = features.float()

        genders = self.genders[index]

        return features, transcript, genders, target


class LastLayerEmbeddingTextCollate:
    def __init__(
        self,
        padding_value: float = 0.0,
        text_tokenizer = None,
    ):
        self.text_tokenizer = text_tokenizer
        self.padding_value = padding_value

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        features, transcripts, genders, targets = zip(*batch)
        targets = torch.stack([torch.as_tensor(t) for t in targets])
        genders = torch.stack([torch.as_tensor(g) for g in genders])

        processed_features = []
        for feature in features:
            # Only check and fix shape if it's (1, T, F)
            if feature.dim() == 3 and feature.shape[0] == 1:
                feature = feature.squeeze(0)  # (T, F)
            processed_features.append(feature)

        features = processed_features
        lengths = [f.shape[0] for f in features]
        max_length = max(lengths)

        feature_dim = features[0].shape[-1]

        padded_features = torch.full((len(features), max_length, feature_dim), self.padding_value)
        for i, f in enumerate(features):
            padded_features[i, :f.shape[0]] = f

        tokenized_transcripts = self.text_tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        return (padded_features, tokenized_transcripts, genders), targets


class LastLayerEmbeddingTextSpeakerEmbDataset(LastLayerEmbeddingTextDataset):
    def __init__(
        self,
        speakeremb_base_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        """Initialization"""
        self.speakeremb_base_dir = speakeremb_base_dir

    def __getitem__(self, index: int) -> Dict[torch.Tensor, torch.Tensor]:
        filename = self.filenames[index]
        if filename.endswith(".wav"):
            filename = filename[:-4] + ".pt"
        filepath = os.path.join(self.base_dir, filename)

        target = self.targets[index]
        transcript = self.transcripts[index]

        # Apply text augmentation
        if self.use_text_augmentation and self.data_type == "train" and random.random() < self.text_augmentation_p:
            transcript = self.text_augmenter.augment(transcript)[0]

        features = self._load_file(filepath)

        # Convert features to float
        features = features.float()

        speaker_emb = torch.load(os.path.join(self.speakeremb_base_dir, filename))
        if speaker_emb.dim() == 2:
            speaker_emb = speaker_emb.squeeze(0)

        return features, transcript, speaker_emb, target


class LastLayerEmbeddingTextSpeakerEmbCollate:
    def __init__(
        self,
        padding_value: float = 0.0,
        text_tokenizer = None,
    ):
        self.text_tokenizer = text_tokenizer
        self.padding_value = padding_value

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        features, transcripts, speaker_embs, targets = zip(*batch)
        targets = torch.stack([torch.as_tensor(t) for t in targets])

        speaker_embs = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in speaker_embs])

        processed_features = []
        for feature in features:
            # Only check and fix shape if it's (1, T, F)
            if feature.dim() == 3 and feature.shape[0] == 1:
                feature = feature.squeeze(0)  # (T, F)
            processed_features.append(feature)

        features = processed_features
        lengths = [f.shape[0] for f in features]
        max_length = max(lengths)

        feature_dim = features[0].shape[-1]

        padded_features = torch.full((len(features), max_length, feature_dim), self.padding_value)
        for i, f in enumerate(features):
            padded_features[i, :f.shape[0]] = f

        tokenized_transcripts = self.text_tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        return (padded_features, tokenized_transcripts, speaker_embs), targets


class LastLayerEmbeddingTextSpeakerEmbMelSpecDataset(LastLayerEmbeddingTextDataset):
    def __init__(
        self,
        target_sr: int = 16000,
        audio_base_dir: str = None,
        speakeremb_base_dir: str = None,
        # Random truncation parameters (Optional)
        use_rand_truncation: Optional[bool] = False,
        min_duration: Optional[float] = 0.0,
        # background noise parameters (Optional)
        use_background_noise: Optional[bool] = False,
        background_noise_dir: Optional[str] = None,
        background_noise_min_snr_in_db: Optional[float] = 3.0,
        background_noise_max_snr_in_db: Optional[float] = 15.0,
        background_noise_p: Optional[float] = 0.5,
        # impulse response parameters (Optional)
        use_rir: Optional[bool] = False,
        rir_dir: Optional[str] = None,
        rir_p: Optional[float] = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        """Initialization"""
        self.audio_base_dir = audio_base_dir
        self.speakeremb_base_dir = speakeremb_base_dir

        self.target_sr = target_sr

        # Data augmentation parameters
        # Random truncation
        self.use_rand_truncation = use_rand_truncation
        self.min_duration = min_duration
        # Background noise
        self.use_background_noise = use_background_noise
        if self.use_background_noise:
            self.background_noise = AddBackgroundNoise(
                background_paths=background_noise_dir,
                min_snr_in_db=background_noise_min_snr_in_db,
                max_snr_in_db=background_noise_max_snr_in_db,
                sample_rate=target_sr,
                target_rate=target_sr,
                p=background_noise_p
            )
        else:
            # Identity augmentation if not using background noise
            self.background_noise = Identity()

        # Impulse response
        self.use_rir = use_rir
        if self.use_rir:
            self.impulse_response = ApplyImpulseResponse(
                ir_paths=rir_dir,
                sample_rate=target_sr,
                target_rate=target_sr,
                p=rir_p,
            )
        else:
            # Identity augmentation if not using impulse response
            self.impulse_response = Identity()

        # Cache for sampling rate resamplers
        self.resamplers = {}


    def _random_truncation(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Cut or pad an audio tensor to the desired length.
        """
        min_length = int(self.min_duration * self.target_sr)
        len_audio = audio.shape[-1]

        if self.use_rand_truncation and len_audio > min_length and random.random() < 0.5:
            segment_length = random.randint(min_length, len_audio)

            max_start = len_audio - segment_length
            start = random.randint(0, max_start)
            end = start + segment_length

            audio = audio[..., start:end]

        return audio

    def _load_wav(self, filepath: str):
        waveform, source_sr = torchaudio.load(filepath)

        # Convert to mono if stereo
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if source_sr != self.target_sr:
            if source_sr not in self.resamplers:
                self.resamplers[source_sr] = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=self.target_sr)
            waveform = self.resamplers[source_sr](waveform)

        return waveform, self.target_sr

    def __getitem__(self, index: int) -> Dict[torch.Tensor, torch.Tensor]:
        filename = self.filenames[index]
        if filename.endswith(".wav"):
            feature_filename = filename[:-4] + ".pt"
        else:
            feature_filename = filename
        filepath = os.path.join(self.base_dir, feature_filename)

        target = self.targets[index]
        transcript = self.transcripts[index]

        # Apply text augmentation
        if self.use_text_augmentation and self.data_type == "train" and random.random() < self.text_augmentation_p:
            transcript = self.text_augmenter.augment(transcript)[0]

        features = self._load_file(filepath)

        # Convert features to float
        features = features.float()

        speaker_emb = torch.load(os.path.join(self.speakeremb_base_dir, feature_filename))
        if speaker_emb.dim() == 2:
            speaker_emb = speaker_emb.squeeze(0)

        audio, _ = self._load_wav(os.path.join(self.audio_base_dir, filename))
        # Random Truncation
        if self.use_rand_truncation and self.data_type == "train":
            audio = self._random_truncation(audio)
        # Background noise insertion or Identity
        audio = self.background_noise(audio.unsqueeze(0)).squeeze(0)
        # Impulse response (Reverberation) or Identity
        audio = self.impulse_response(audio.unsqueeze(0)).squeeze(0)

        return features, transcript, speaker_emb, audio, target


class LastLayerEmbeddingTextSpeakerEmbMelSpecCollate:
    def __init__(
        self,
        padding_value: float = 0.0,
        text_tokenizer = None,
    ):
        self.text_tokenizer = text_tokenizer
        self.padding_value = padding_value

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        features, transcripts, speaker_embs, audios, targets = zip(*batch)
        targets = torch.stack([torch.as_tensor(t) for t in targets])

        speaker_embs = torch.stack([t if isinstance(t, torch.Tensor) else torch.tensor(t) for t in speaker_embs])

        processed_features = []
        for feature in features:
            # Only check and fix shape if it's (1, T, F)
            if feature.dim() == 3 and feature.shape[0] == 1:
                feature = feature.squeeze(0)  # (T, F)
            processed_features.append(feature)

        features = processed_features
        lengths = [f.shape[0] for f in features]
        max_length = max(lengths)

        feature_dim = features[0].shape[-1]

        padded_features = torch.full((len(features), max_length, feature_dim), self.padding_value)
        for i, f in enumerate(features):
            padded_features[i, :f.shape[0]] = f

        tokenized_transcripts = self.text_tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # pad audios ana take the length
        audios_lengths = [audio.shape[-1] for audio in audios]
        # pad audios
        max_length = max(audios_lengths)

        padded_audios = torch.full((len(audios), max_length), self.padding_value)

        for i, audio in enumerate(audios):
            padded_audios[i, :audio.shape[-1]] = audio.float()

        return (padded_features, tokenized_transcripts, speaker_embs, padded_audios), targets


class LastLayerEmbeddingTextMelSpecDataset(LastLayerEmbeddingTextDataset):
    def __init__(
        self,
        target_sr: int = 16000,
        audio_base_dir: str = None,
        # Random truncation parameters (Optional)
        use_rand_truncation: Optional[bool] = False,
        min_duration: Optional[float] = 0.0,
        # background noise parameters (Optional)
        use_background_noise: Optional[bool] = False,
        background_noise_dir: Optional[str] = None,
        background_noise_min_snr_in_db: Optional[float] = 3.0,
        background_noise_max_snr_in_db: Optional[float] = 15.0,
        background_noise_p: Optional[float] = 0.5,
        # impulse response parameters (Optional)
        use_rir: Optional[bool] = False,
        rir_dir: Optional[str] = None,
        rir_p: Optional[float] = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        """Initialization"""
        self.audio_base_dir = audio_base_dir
        self.target_sr = target_sr

        # Data augmentation parameters
        # Random truncation
        self.use_rand_truncation = use_rand_truncation
        self.min_duration = min_duration
        # Background noise
        self.use_background_noise = use_background_noise
        if self.use_background_noise:
            self.background_noise = AddBackgroundNoise(
                background_paths=background_noise_dir,
                min_snr_in_db=background_noise_min_snr_in_db,
                max_snr_in_db=background_noise_max_snr_in_db,
                sample_rate=target_sr,
                target_rate=target_sr,
                p=background_noise_p
            )
        else:
            # Identity augmentation if not using background noise
            self.background_noise = Identity()

        # Impulse response
        self.use_rir = use_rir
        if self.use_rir:
            self.impulse_response = ApplyImpulseResponse(
                ir_paths=rir_dir,
                sample_rate=target_sr,
                target_rate=target_sr,
                p=rir_p,
            )
        else:
            # Identity augmentation if not using impulse response
            self.impulse_response = Identity()

        # Cache for sampling rate resamplers
        self.resamplers = {}


    def _random_truncation(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Cut or pad an audio tensor to the desired length.
        """
        min_length = int(self.min_duration * self.target_sr)
        len_audio = audio.shape[-1]

        if self.use_rand_truncation and len_audio > min_length and random.random() < 0.5:
            segment_length = random.randint(min_length, len_audio)

            max_start = len_audio - segment_length
            start = random.randint(0, max_start)
            end = start + segment_length

            audio = audio[..., start:end]

        return audio

    def _load_wav(self, filepath: str):
        waveform, source_sr = torchaudio.load(filepath)

        # Convert to mono if stereo
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if source_sr != self.target_sr:
            if source_sr not in self.resamplers:
                self.resamplers[source_sr] = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=self.target_sr)
            waveform = self.resamplers[source_sr](waveform)

        return waveform, self.target_sr

    def __getitem__(self, index: int) -> Dict[torch.Tensor, torch.Tensor]:
        filename = self.filenames[index]
        if filename.endswith(".wav"):
            feature_filename = filename[:-4] + ".pt"
        else:
            feature_filename = filename
        filepath = os.path.join(self.base_dir, feature_filename)

        target = self.targets[index]
        transcript = self.transcripts[index]

        # Apply text augmentation
        if self.use_text_augmentation and self.data_type == "train" and random.random() < self.text_augmentation_p:
            transcript = self.text_augmenter.augment(transcript)[0]

        features = self._load_file(filepath)

        # Convert features to float
        features = features.float()

        audio, _ = self._load_wav(os.path.join(self.audio_base_dir, filename))
        # Random Truncation
        if self.use_rand_truncation and self.data_type == "train":
            audio = self._random_truncation(audio)

        if self.data_type == "train":
            # Background noise insertion or Identity
            audio = self.background_noise(audio.unsqueeze(0)).squeeze(0)
            # Impulse response (Reverberation) or Identity
            audio = self.impulse_response(audio.unsqueeze(0)).squeeze(0)

        genders = self.genders[index]

        return features, transcript, genders, audio, target


class LastLayerEmbeddingTextMelSpecCollate:
    def __init__(
        self,
        padding_value: float = 0.0,
        text_tokenizer = None,
    ):
        self.text_tokenizer = text_tokenizer
        self.padding_value = padding_value

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        features, transcripts, genders, audios, targets = zip(*batch)
        targets = torch.stack([torch.as_tensor(t) for t in targets])

        genders = torch.stack([torch.as_tensor(g) for g in genders])

        processed_features = []
        for feature in features:
            # Only check and fix shape if it's (1, T, F)
            if feature.dim() == 3 and feature.shape[0] == 1:
                feature = feature.squeeze(0)  # (T, F)
            processed_features.append(feature)

        features = processed_features
        lengths = [f.shape[0] for f in features]
        max_length = max(lengths)

        feature_dim = features[0].shape[-1]

        padded_features = torch.full((len(features), max_length, feature_dim), self.padding_value)
        for i, f in enumerate(features):
            padded_features[i, :f.shape[0]] = f

        tokenized_transcripts = self.text_tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # pad audios ana take the length
        audios_lengths = [audio.shape[-1] for audio in audios]
        # pad audios
        max_length = max(audios_lengths)

        padded_audios = torch.full((len(audios), max_length), self.padding_value)

        for i, audio in enumerate(audios):
            padded_audios[i, :audio.shape[-1]] = audio.float()

        return (padded_features, tokenized_transcripts, genders, padded_audios), targets