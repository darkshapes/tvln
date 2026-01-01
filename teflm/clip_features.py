# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from enum import Enum
from typing import Literal

from open_clip import list_pretrained
from torch import Tensor


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


def get_model_and_pretrained(member: ModelType) -> tuple[str, str]:
    """Return the raw strings for a member.\n

    :param member: Enum member representing a model and its pretrained variant.
    :returns: The model type and pretrained string."""

    return member.value


class CLIPFeatures:
    """Convenience wrapper around the Open‑CLIP model for image feature extraction."""

    def __init__(self) -> None:
        """Create a CLIPFeatures instance with the default model configuration (VIT_L_14_LAION2B_S32B_B82K @ FP32)."""
        model_name, dataset_name = get_model_and_pretrained(ModelType.VIT_L_14_LAION2B_S32B_B82K)
        self._model_type: str = model_name
        self._pretrained: str = dataset_name
        self._precision: str = "fp32"

    def ImageEncoder(self, images) -> Tensor:
        """Encode a batch of images into CLIP features.\n
        :param images: Paths to the image files.
        :returns Concatenated image feature vectors."""

        from open_clip import create_model_and_transforms
        from PIL import Image
        from torch import cat as torch_cat, no_grad as torch_no_grad, stack as torch_stack

        vlmodel, preprocess_train, feature_extractor = create_model_and_transforms(
            self._model_type,
            pretrained=self._pretrained,
            precision=self._precision,
            device=self._device,
        )

        batch_size = 512
        image_features_list = []

        for i in range(0, len(self.images), batch_size):
            batch_images = self.images[i : i + batch_size]
            image_inputs = torch_stack([preprocess_train(Image.open(img).convert("RGB")) for img in batch_images])

            with torch_no_grad():
                batch_image_features = vlmodel.encode_image(image_inputs)
            image_features_list.append(batch_image_features)

        image_features = torch_cat(image_features_list, dim=0)
        return image_features

    def _set_images(self, image_paths: list[str] | str) -> None:
        """Internal helper to normalise the ``images`` attribute.\n
        :param image_paths: One or more image file paths."""

        if isinstance(image_paths, list):
            self.images: list[str] = image_paths
        else:
            self.images = [image_paths]

    def set_device(self, device_name: DeviceName) -> None:
        """Set the computation device.\n
        :param device_name : Target graphics processing device."""
        self._device: str = device_name.value

    def set_model_type(self, model_type: ModelType) -> None:
        """Switch the underlying Open‑CLIP model.
        :param model_type: Desired pretrained model and dataset variant."""
        model_name, pretrained = get_model_and_pretrained(model_type)
        self._model_type = model_name
        self._pretrained = pretrained

    def set_precision(self, precision: PrecisionType) -> None:
        """Change the numeric precision used by the model.
        :param precision: Desired float calculation precision."""
        self._precision = precision.value

    def extract(self, image_paths) -> Tensor:
        """Convenience entry‑point that sets images and returns CLIP features.\n
        :param image_paths: One or more image file paths.
        :returns: Extracted image features"""
        self._set_images(image_paths)
        return self.ImageEncoder(self.images)
