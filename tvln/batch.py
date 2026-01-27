# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

import torch


class ImageFile:
    _image_path: str

    def __init__(self) -> None:
        """Initializes an ImageFile instance with a default"""
        self._default_path: Path = Path(__file__).resolve().parent / "assets" / "DSC_0047.png"
        self._default_path.resolve()
        self._default_path.as_posix()
        self._image_path = ""

    def single_image(self) -> None:
        """Set absolute path to an image file, ensuring the file exists, falling back to a default image if none is provided."""
        from sys import modules as sys_modules

        if not self.image_path and "pytest" not in sys_modules:
            image_path = input("Enter the path to an image file (e.g. /home/user/image.png, C:/Users/user/Pictures/...): ")
        else:
            image_path = None
        if not image_path:
            image_path = self._default_path
        if not Path(image_path).resolve().is_file():
            raise FileNotFoundError(f"File not found: {image_path}")
        else:
            image_path = Path(image_path).resolve()
        self._image_path = str(image_path.as_posix())
        if not isinstance(self._image_path, str):
            raise TypeError(f"Expected a string or list of strings for `image_paths` {self._image_path}, got {type(self._image_path)} ")

    def as_tensor(self, dtype: torch.dtype, device: str, normalize: bool = False) -> None:
        """Convert a Pillow `Image` to a batched `torch.Tensor`\n
        :param image: Pillow image (RGB) to encode.
        :param device: Target device for the tensor (default: ``gpu.device``).
        :param normalize:  Normalize tensor to [-1, 1]:
        :return: Tensor of shape ``[1, 3, H, W]`` on ``device``."""

        from numpy import array as np_array
        from numpy._typing import NDArray
        from PIL.Image import open as open_img

        with open_img(str(self._image_path)).convert("RGB") as pil_image:
            numeric_image: NDArray = np_array(pil_image).astype("float32") / 255.0  # HWC, 0‑1
            numeric_image: NDArray = numeric_image.transpose(2, 0, 1)  # CHW
            tensor = torch.from_numpy(numeric_image).unsqueeze(0).to(dtype=dtype, device=device)
            if normalize:
                tensor = tensor * 2.0 - 1.0
            self.tensor = tensor

    @property
    def image_path(self) -> str:
        """Reveal the current image path"""
        return self._image_path

    def set_image_path(self, image) -> None:
        """Change the current image path"""
        self._image_path = image


# ... existing imports ...


def batch_process_images(image_paths: Iterable[str], extractor: Callable, device: str) -> dict[str, torch.Tensor]:
    """Process many images with a single FeatureExtractor instance.\n
    :param image_paths: Paths to images.\n
    :param model: The model to use for extraction.\n
    :returns: Mapping of image paths to feature tensors.\n
    :raises FileNotFoundError: If an image does not exist.\n
    :raises ValueError: If no image paths are supplied."""
    from tqdm import tqdm

    if not image_paths:
        raise ValueError("No image paths supplied")

    image_file = ImageFile()

    features: dict[str, torch.Tensor] = {}
    for path in tqdm(image_paths, desc="processing_images..."):
        image_file.set_image_path(path)
        image_file.as_tensor(device=device, dtype=torch.float32, normalize=False)
        extractor.set_image_file(image_file)
        tensor = extractor._extract_vae()
        features[path] = tensor
    return features
