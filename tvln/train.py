import torch
from tvln.datasets import build_datasets, process_dataset
from tvln.extract import FeatureExtractor
from tvln.batch import batch_process_images, ImageFile
from tvln.options import DeviceName


def main():
    device = DeviceName.CPU
    if torch.cuda.is_available():
        device = DeviceName.CUDA
    elif torch.mps.is_available():
        device = DeviceName.MPS
    datasets = build_datasets()

    synthetic_data = datasets["synthetic"]
    original_data = datasets["original"]

    model = "black-forest-labs/FLUX.1-dev"
    image_file = ImageFile()
    extractor = FeatureExtractor(image_file)
    extractor.set_model(model)

    synthetic_paths = [image_data["image"]["path"] for image_data in synthetic_data]  # type: ignore

    original_paths = [image_data["image"]["path"] for image_data in original_data]  # type: ignore

    synthetic_features = batch_process_images(image_paths=synthetic_paths, extractor=extractor, device=device)  # type: ignore
    original_features = batch_process_images(image_paths=original_paths, extractor=extractor, device=device)  # type: ignore


if __name__ == "__main__":
    main()
