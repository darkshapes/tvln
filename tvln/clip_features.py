# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from enum import Enum

from open_clip import list_pretrained
from open_clip.pretrained import _PRETRAINED
from torch import Tensor, nn

from tvln.batch import ImageFile


class DeviceName(str, Enum):
    """Graphics processors usable by the CLIP pipeline."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class PrecisionType(str, Enum):
    """Supported numeric float precision."""

    FP64 = "fp64"
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"


ModelType = Enum(
    "ModelData",
    {
        # member name → (model_type, pretrained) value
        f"{model.replace('-', '_').upper()}_{pretrained.replace('-', '_').upper()}": (
            model,
            pretrained,
        )
        for model, pretrained in list_pretrained()
    },
)


ModelLink = Enum(
    "ModelData",
    {
        f"{family.replace('-', '_').upper()}_{id.replace('-', '_').upper()}": (data.get("hf_hub", "").strip("/"), data.get("url"))
        for family, name in _PRETRAINED.items()
        for id, data in name.items()
        if data.get("hf_hub") or data.get("url")
    },
)


def get_model_and_pretrained(member: Enum) -> tuple[str, str]:
    """Return the raw strings for a member.\n
    :param member: Enum member representing a model and its pretrained variant.
    :returns: The model type and pretrained string."""
    return member.value


class CLIPEncoder(nn.Module):
    """CLIP wrapper courtesy ncclab-sustech/BrainFLORA"""

    def __init__(self, device: str = "cpu", model: str = "openai/clip-vit-large-patch14") -> None:
        """Instantiate the encoder with a specific device and model\n
        :param device: The graphics device to allocate, Default is cpu"""
        from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize
        from transformers import CLIPVisionModel

        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(model).to(device)
        self.clip_size = (224, 224)
        self.preprocess = Compose(
            [
                Resize(size=self.clip_size[0], interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size=self.clip_size),
                Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        self._device = device

    def clip_encode_image(self, x: Tensor) -> Tensor:
        """Encode image patches using CLIP vision model\n
        Include class and positional embedding, then stop at second-to-last layer where features are strongest\n
        :param x: Tensors of the image being processed"""

        import torch

        x = x.reshape(x.shape[0], x.shape[1], -1)  # [batchsize, 1024, 256]
        x = x.permute(0, 2, 1)

        class_embedding = self.clip.vision_model.embeddings.class_embedding.to(x.dtype)
        class_embedding = class_embedding.repeat(x.shape[0], 1, 1)  # [batchsize, 1, 1024]
        x = torch.cat([class_embedding, x], dim=1)

        pos_embedding = self.clip.vision_model.embeddings.position_embedding
        position_ids = torch.arange(0, 257).unsqueeze(0).to(self._device)
        x = x + pos_embedding(position_ids)

        x = self.clip.vision_model.pre_layrnorm(x)
        x = self.clip.vision_model.encoder(x, output_hidden_states=True)

        select_hidden_state_layer = -2
        select_hidden_state = x.hidden_states[select_hidden_state_layer]  # [1, 256, 1024]
        image_features = select_hidden_state[:, 1:]  # Remove class token

        return image_features

    def encode_image(self, x: Tensor) -> Tensor:
        """Full image encoding pipeline
        :param x: the input image tensor in shape [B, C, H, W] and device-compatible dtype."""
        x = x.to(self._device)
        x = self.preprocess(x)  # [3, 224, 224]
        x = self.clip.vision_model.embeddings.patch_embedding(x)  # [1024, 16, 16]
        image_feats = self.clip_encode_image(x)
        return image_feats


class CLIPFeatures:
    """Convenience wrapper around the Open-CLIP model for image feature extraction."""

    def __init__(self) -> None:
        """Create a CLIPFeatures instance with the default model configuration (VIT_L_14_LAION2B_S32B_B82K @ FP32)."""
        self._images = []
        model_name, dataset_name = get_model_and_pretrained(ModelType.VIT_L_14_LAION2B_S32B_B82K)  # type:ignore
        self._model_type: str = model_name
        self._pretrained: str = dataset_name
        self._precision: str = "fp32"

    def ImageEncoder(self) -> Tensor:
        """Encode a batch of images into CLIP features.\n
        :param images: Paths to the image files.
        :returns Concatenated image feature vectors."""

        from open_clip import create_model_and_transforms
        from PIL import Image
        from torch import cat as torch_cat
        from torch import no_grad as torch_no_grad
        from torch import stack as torch_stack

        vlmodel, preprocess_train, feature_extractor = create_model_and_transforms(
            self._model_type,
            pretrained=self._pretrained,
            precision=self._precision,
            device=self._device,
        )

        batch_size = 512
        image_features_list = []

        for i in range(0, len(self._images), batch_size):
            batch_images = self._images[i : i + batch_size]
            image_inputs = torch_stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images])  # type:ignore

            with torch_no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)
            image_features_list.append(batch_image_features)

        image_features = torch_cat(image_features_list, dim=0)
        return image_features

    def set_device(self, device_name: DeviceName) -> None:
        """Set the computation device.\n
        :param device_name : Target graphics processing device."""
        self._device: str = device_name.value

    def set_model_type(self, model_type: Enum) -> None:
        """Switch the underlying Open-CLIP model.
        :param model_type: Desired pretrained model and dataset variant."""
        model_name, pretrained = get_model_and_pretrained(model_type)
        self._model_type = model_name
        self._pretrained = pretrained

    def set_model_link(self, model_link: Enum) -> None:
        """Switch the path to an Open-CLIP model
        :param model_link: Desired pretrained model and dataset variant."""
        model_link, model_hub = get_model_and_pretrained(model_link)
        self._model_link: str = model_link
        self._model_hub: str = model_hub

    def set_precision(self, precision: PrecisionType) -> None:
        """Change the numeric precision used by the model.
        :param precision: Desired float calculation precision."""
        self._precision = precision.value

    def extract(self, image: ImageFile, last_layer=False) -> Tensor:
        """Convenience entry-point that sets images and returns CLIP features.\n
        :param image_paths: One or more image file paths.
        :returns: Extracted image features"""
        if not last_layer:
            clip_encoder = CLIPEncoder(self._device, model=self._model_link)

            return clip_encoder.encode_image(image.tensor)
        else:
            self._images = [image._image_path]
            return self.ImageEncoder()
