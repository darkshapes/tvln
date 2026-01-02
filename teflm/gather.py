# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import torch
from pathlib import Path


class Gather:
    _image_path: str

    def __init__(self) -> None:
        self._default_path = Path(__file__).resolve().parent / "assets" / "DSC_0047.png"
        self._default_path.resolve()
        self._default_path.as_posix()

    def single_image(self) -> None:
        image_path = input("Enter the path to an image file (e.g. /home/user/image.png, C:/Users/user/Pictures/...): ")
        if not image_path:
            image_path = self._default_path
        if not Path(image_path).resolve().is_file():
            raise FileNotFoundError(f"File not found: {image_path}")
        else:
            image_path = Path(image_path).resolve()
        self._image_path = str(image_path.as_posix())
        if not isinstance(self._image_path, str):
            raise TypeError(f"Expected a string or list of strings for `image_paths` {self._image_path}, got {type(self._image_path)} ")

    def as_tensor(self, dtype: torch.dtype, device: torch.device, normalize: bool = False) -> None:
        """Convert a Pillow `Image` to a batched `torch.Tensor`\n
        :param image: Pillow image (RGB) to encode.
        :param device: Target device for the tensor (default: ``gpu.device``).
        :param normalize:  Normalize tensor to [-1, 1]:
        :return: Tensor of shape ``[1, 3, H, W]`` on ``device``."""

        from numpy._typing import NDArray
        from numpy import array as np_array
        from PIL.Image import open as open_img

        with open_img(str(self._image_path)).convert("RGB") as pil_image:
            numeric_image: NDArray = np_array(pil_image).astype("float32") / 255.0  # HWC, 0‑1
            numeric_image: NDArray = numeric_image.transpose(2, 0, 1)  # CHW
            tensor = torch.from_numpy(numeric_image).unsqueeze(0).to(dtype=dtype, device=device)
            if normalize:
                tensor = tensor * 2.0 - 1.0
            self.tensor = tensor
