# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


def main():
    import os
    from pprint import pprint
    from teflm.clip_features import CLIPFeatures, DeviceName, PrecisionType, ModelType
    from teflm.gather import Gather
    from huggingface_hub import snapshot_download
    from diffusers import AutoencoderKL
    import torch

    text_device = DeviceName.CPU
    if torch.cuda.is_available():
        device = DeviceName.CUDA
    elif torch.mps.is_available():
        device = DeviceName.MPS
    precision = PrecisionType.FP32

    feature_extractor = CLIPFeatures()
    feature_extractor.set_device(text_device)
    feature_extractor.set_precision(precision)

    gather = Gather()
    image_path = gather.single_image()

    clip_l_tensor = feature_extractor.extract(image_path)
    clip_l_data = vars(feature_extractor)

    feature_extractor = CLIPFeatures()
    feature_extractor.set_device(text_device)
    feature_extractor.set_precision(precision)
    feature_extractor.set_model_type(ModelType.VIT_L_14_LAION400M_E32)
    clip_l_e32_tensor = feature_extractor.extract(image_path)
    clip_l_e32_data = vars(feature_extractor)

    feature_extractor = CLIPFeatures()
    feature_extractor.set_device(text_device)
    feature_extractor.set_precision(precision)
    feature_extractor.set_model_type(ModelType.VIT_BIGG_14_LAION2B_S39B_B160K)
    clip_g_tensor = feature_extractor.extract(image_path)
    clip_g_data = vars(feature_extractor)

    vae_path = snapshot_download("black-forest-labs/FLUX.1-dev", allow_patterns=["vae/*"])
    vae_path = os.path.join(vae_path, "vae")

    vae_model = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16).to(device)

    image_tensor = gather.as_tensor(dtype=torch.float16, device=device)
    vae_tensor = vae_model.encode(image_tensor)

    aggregate_data = {
        "CLIP_L_S32B82K": [clip_l_data, clip_l_tensor],
        "CLIP_L_400M_E32": [clip_l_e32_data, clip_l_e32_tensor],
        "CLIP_BIG_G_S39B_B160K": [clip_g_data, clip_g_tensor],
        "F1 VAE": ["black-forest-labs/FLUX.1-dev", vae_tensor.values],
    }
    pprint(aggregate_data)


if __name__ == "__main__":
    main()
