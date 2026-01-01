# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->


def main():
    from pprint import pprint
    from pathlib import Path
    from teflm.clip_features import CLIPFeatures, DeviceName, PrecisionType, ModelType

    feature_extractor = CLIPFeatures()
    feature_extractor.set_device(DeviceName.CPU)
    feature_extractor.set_precision(PrecisionType.FP32)

    image_path = Path(__file__).resolve().parent / "assets" / "DSC_0047.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"File not found: {image_path}")

    clip_l_tensor = feature_extractor.extract(image_path)
    feature_extractor.set_model_type(ModelType.VIT_L_14_LAION400M_E32)
    clip_l_e32_tensor = feature_extractor.extract(image_path)
    feature_extractor.set_model_type(ModelType.VIT_BIGG_14_LAION2B_S39B_B160K)
    clip_g_tensor = feature_extractor.extract(image_path)
    pprint({"CLIP_L_S32B82K": clip_l_tensor, "CLIP_L_400M_E32": clip_l_e32_tensor, "CLIP_BIG_G_S39B_B160K": clip_g_tensor})


if __name__ == "__main__":
    main()
