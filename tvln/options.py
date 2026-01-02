# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


from enum import Enum
from open_clip import list_pretrained
from open_clip.pretrained import _PRETRAINED


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
    FLOAT16 = "torch.float16"
    BFLOAT16 = "torch.bfloat16"
    FLOAT32 = "torch.float32"
    FLOAT64 = "torch.float64"


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
