# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, Image, IterableDataset, interleave_datasets, load_dataset

from tvln.batch import ImageFile
from tvln.extract import FeatureExtractor
from tvln.options import DeviceName


def build_datasets() -> dict[str, Dataset | dict[str, IterableDataset] | IterableDataset | DatasetDict]:
    """Builds synthetic and original datasets.\n
    :returns: A dictionary containing synthetic and original datasets."""

    synthetic_input_folder = ".datasets"
    original_input_folder = Path(__file__).parent / "assets" / "ph"

    slice_dataset = load_dataset("darkshapes/a_slice", cache_dir=str(synthetic_input_folder), split="train").cast_column("image", Image(decode=False))
    rnd_synthetic_dataset = load_dataset("exdysa/rnd_synthetic_img", cache_dir=str(synthetic_input_folder), split="train").cast_column("image", Image(decode=False))

    synthetic_dataset = interleave_datasets([slice_dataset, rnd_synthetic_dataset])
    original_folder_contents = [{"image": str(image)} for image in original_input_folder.iterdir() if image.is_file()]
    original_dataset = Dataset.from_list(original_folder_contents).cast_column("image", Image(decode=False))
    return {"synthetic": synthetic_dataset, "original": original_dataset}


@torch.no_grad
async def process_dataset(dataset) -> dict[str, torch.Tensor]:
    """Processes a dataset to extract features.\n
    :param dataset: The dataset to process.
    :returns: A dictionary mapping image paths to their feature tensors.
    :raises ValueError: If the dataset is empty."""

    device = DeviceName.CPU
    if torch.cuda.is_available():
        device = DeviceName.CUDA
    elif torch.mps.is_available():
        device = DeviceName.MPS

    features = {}
    image_file = ImageFile()
    for image_data in dataset:
        image_path = image_data["image"]["path"]
        image_file.set_image_path(image_path)
        image_file.single_image()
        image_file.as_tensor(device=device, dtype=torch.float32)
        feature_extractor = FeatureExtractor(image=image_file)
        vae_tensor, _ = feature_extractor.extract(model="black-forest-labs/FLUX.1-dev")
        features.setdefault(image_path, vae_tensor)
    return features
