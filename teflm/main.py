# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path

from nnll.init_gpu import Gfx
from nnll.embeds import load_image_as_tensor, HFEmbedder

gfx = Gfx(full_precision=True)


def main():
    clip: HFEmbedder = HFEmbedder("openai/clip-vit-large-patch14", max_length=512, dtype=gfx.dtype).to(gfx.device)
    tfive: HFEmbedder = HFEmbedder("google/t5-v1_1-xxl", max_length=77, dtype=gfx.dtype).to(gfx.device)

    image_path = Path(__file__).resolve().parent / "assets" / "DSC_0047.png"

    # 2. Check existence
    if not image_path.is_file():
        raise FileNotFoundError(f"File not found: {image_path}")

    from torch import Tensor

    image_tensor = load_image_as_tensor(image_path, gfx.dtype, device=gfx.device)

    print(image_tensor)


if __name__ == "__main__":
    main()
