# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from enum import Enum

import torch
from tvln.batch import ImageFile
from tvln.clip_features import CLIPFeatures
from tvln.options import DeviceName, ModelLink, ModelType, PrecisionType
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from huggingface_hub import snapshot_download


class FeatureExtractor:
    def __init__(self, device: DeviceName, precision: PrecisionType, image_file: ImageFile):
        self.device = device
        self.precision = precision
        self.image_file = image_file

    def extract_features(self, model_info: Enum | str, last_layer: bool = False):
        """Extract features from the image using the specified model.
        :param model_info: The kind of model to use
        :param last_layer: Whether the features are extracted from the last layer of the model or from an intermediate layer."""

        if isinstance(model_info, str):
            import os

            vae_path = snapshot_download(model_info, allow_patterns=["vae/*"])
            vae_path = os.path.join(vae_path, "vae")
            vae_model = AutoencoderKL.from_pretrained(vae_path, torch_dtype=self.precision).to(self.device.value)
            vae_tensor = vae_model.tiled_encode(self.image_file.tensor, return_dict=False)
            return vae_tensor, model_info
        clip_extractor = CLIPFeatures()
        clip_extractor.set_device(self.device)
        clip_extractor.set_precision(self.precision)
        if isinstance(model_info, ModelLink):
            clip_extractor.set_model_link(model_info)
        elif isinstance(model_info, ModelType):
            clip_extractor.set_model_type(model_info)
        print(clip_extractor._precision)
        tensor = clip_extractor.extract(self.image_file, last_layer)
        data = vars(clip_extractor)
        self.cleanup(model=clip_extractor)
        return tensor, data

    def cleanup(self, model: CLIPFeatures) -> None:  # type:ignore
        """Cleans up the model and frees GPU memory
        :param model: The model instance used for feature extraction"""

        import gc

        import torch

        if self.device != "cpu":
            gpu = getattr(torch, self.device)
            gpu.empty_cache()
        model: None = None
        del model
        gc.collect()

    def set_device(self, device: DeviceName):
        self.device = device

    def set_precision(self, precision: PrecisionType | torch.dtype):
        if isinstance(precision, PrecisionType):
            self.precision = precision.value
        else:
            self.precision = precision
