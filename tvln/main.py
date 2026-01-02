# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import torch


@torch.no_grad
def main():
    from tvln.batch import ImageFile
    from tvln.options import DeviceName, ModelLink, ModelType, PrecisionType
    from tvln.extract import FeatureExtractor

    text_device: str = DeviceName.CPU
    device = text_device
    precision = PrecisionType.FP32
    dtype = torch.float32
    if torch.cuda.is_available():
        device = DeviceName.CUDA
        precision = PrecisionType.FP32
    elif torch.mps.is_available():
        device = DeviceName.MPS
        precision = PrecisionType.FP32

    image_file = ImageFile()
    image_file.single_image()

    feature_extractor = FeatureExtractor(text_device, precision=precision, image_file=image_file)
    print(feature_extractor.precision)
    clip_l_tensor, clip_l_data = feature_extractor.extract_features(model_info=ModelLink.VIT_L_14_LAION2B_S32B_B82K)
    print(feature_extractor.precision)
    clip_l_e32_tensor, clip_l_e32_data = feature_extractor.extract_features(ModelType.VIT_L_14_LAION400M_E32, last_layer=True)
    print(feature_extractor.precision)
    clip_g_tensor, clip_g_data = feature_extractor.extract_features(ModelLink.VIT_BIGG_14_LAION2B_S39B_B160K)

    image_file.as_tensor(dtype=dtype, device=device)
    feature_extractor.set_device(device)
    feature_extractor.set_precision(torch.float32)
    vae_tensor, vae_data = feature_extractor.extract_features("black-forest-labs/FLUX.1-dev")

    aggregate_data = {
        "CLIP_L_S32B82K": [clip_l_data, clip_l_tensor],
        "CLIP_L_400M_E32": [clip_l_e32_data, clip_l_e32_tensor],
        "CLIP_BIG_G_S39B_B160K": [clip_g_data, clip_g_tensor],
        "F1 VAE": ["black-forest-labs/FLUX.1-dev", vae_tensor[0].sample()],
    }
    print(aggregate_data)
    return aggregate_data


if __name__ == "__main__":
    main()
