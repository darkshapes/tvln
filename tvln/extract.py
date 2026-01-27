# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from enum import Enum

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from huggingface_hub import snapshot_download

from tvln.batch import ImageFile
from tvln.clip_features import FloraEncoder, OpenClipEncoder
from tvln.options import FloraModel, OpenClipModel


class FeatureExtractor:
    encoder: FloraEncoder | OpenClipEncoder | AutoencoderKL = OpenClipEncoder()

    def __init__(self, image: ImageFile):
        self.image: ImageFile = image

    def extract(self, model: Enum | str | None = None) -> tuple[torch.Tensor, str | dict]:
        """Extract features from the image using the specified model.
        :param model_info: The kind of model to use
        :param image: One or more image file paths.
        :returns: Extracted image features"""

        self.dtype = self.image.tensor.dtype
        self.device = self.image.tensor.device
        self.model = model or self.model

        if isinstance(model, FloraModel):  # type: ignore
            tensor = self._extract_flora()
        elif isinstance(model, OpenClipModel):  # type: ignore
            tensor = self._extract_openclip()
        else:
            tensor = self._extract_vae()

        data: dict = {"model": model, "dtype": self.image.tensor.dtype, "device": self.image.tensor.device} | vars(self.image)
        self.cleanup()
        return tensor, data

    @torch.no_grad
    def _extract_flora(self) -> torch.Tensor:
        """Extract features using a Flora model."""
        if isinstance(self.encoder, FloraEncoder) and self.encoder.flora_model == self.model.value[0]:
            flora_encoder = self.encoder
        else:
            flora_encoder = FloraEncoder(device=self.device.type)
            flora_encoder.flora_model, _ = self.model.value  # type: ignore
        tensor: torch.Tensor = flora_encoder.encode_image(self.image.tensor)
        self.encoder = flora_encoder
        return tensor

    @torch.no_grad
    def _extract_openclip(self) -> torch.Tensor:
        """Extract features using an OpenClip model."""
        if isinstance(self.encoder, OpenClipEncoder) and self.encoder.open_clip_model == self.model.value[0]:
            open_clip_encoder = self.encoder
        else:
            open_clip_encoder = OpenClipEncoder(device=self.device.type, precision=self.dtype)
            open_clip_encoder.open_clip_model, open_clip_encoder.pretraining = self.model.value  # type: ignore
        open_clip_encoder.precision = self.image.tensor.dtype
        tensor: torch.Tensor = open_clip_encoder.encode_image(self.image)
        self.encoder = open_clip_encoder
        return tensor

    @torch.no_grad
    def _extract_vae(self) -> torch.Tensor:
        """Extract features using a VAE model, re‑using the model when possible."""
        if isinstance(self.encoder, AutoencoderKL):
            vae_model = self.encoder
        else:
            import os

            vae_path = snapshot_download(self.model, allow_patterns=["vae/*"])  # type: ignore
            vae_path = os.path.join(vae_path, "vae")
            vae_model = AutoencoderKL.from_pretrained(vae_path, torch_dtype=self.dtype).to(self.device.type)  # type: ignore DeviceLike
            self.encoder = vae_model

        vae_tensor = vae_model.tiled_encode(self.image.tensor, return_dict=False)
        tensor = vae_tensor[0].sample()
        return tensor

    @property
    def model_name(self) -> Enum | str | None:
        """Reveal the current model"""

        return self.model

    def set_model(self, model) -> None:
        """Change the current model"""
        self.model = model

    @property
    def image_file(self) -> ImageFile:
        """Reveal the current image file"""
        return self.image

    def set_image_file(self, image_file: ImageFile) -> None:
        """Change the current image file and align dtypes"""
        self.image = image_file
        self.dtype = self.image.tensor.dtype
        self.device = self.image.tensor.device

    def cleanup(self) -> None:  # type:ignore
        """Cleans up the model and frees GPU memory
        :param model: The model instance used for feature extraction"""

        import gc

        device = self.image.tensor.device.type
        if device != "cpu":
            gpu = getattr(torch, device)
            gpu.empty_cache()
        del self.encoder
        gc.collect()
