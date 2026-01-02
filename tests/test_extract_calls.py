# SPDX-License-Identifier: MPL-2.0 AND LicenseRef-Commons-Clause-License-Condition-1.0
# <!-- // /*  d a r k s h a p e s */ -->

import pytest
from unittest.mock import patch, MagicMock
from tvln.clip_features import CLIPFeatures
from tvln.extract import FeatureExtractor, AutoencoderKL, snapshot_download


@pytest.fixture
def dummy_image_file():
    """Return a minimal ImageFile‑like object."""
    return MagicMock()


@pytest.fixture
def dummy_download():
    """Patch `snapshot_download` so that no network traffic occurs."""
    with patch("tvln.extract.snapshot_download", autospec=True) as mock_dl:
        mock_dl.return_value = "/tmp/vae"  # dummy path
        yield mock_dl


@pytest.fixture
def dummy_encoder():
    """
    Patch `AutoencoderKL` so that the model is never actually loaded.
    The patched class returns a mock instance whose `tiled_encode` method
    yields a dummy tensor.
    """
    with patch("tvln.extract.AutoencoderKL", autospec=True) as MockKL:
        # The classmethod `from_pretrained` should return a mock model
        mock_model = MagicMock(name="vae_model")
        MockKL.from_pretrained.return_value = mock_model
        # The instance method `tiled_encode` returns a dummy tensor
        mock_model.tiled_encode.return_value = MagicMock(name="vae_tensor")
        yield MockKL


@patch.object(FeatureExtractor, "cleanup")
@patch.object(CLIPFeatures, "extract", return_value=MagicMock(name="tensor"))
@patch.object(CLIPFeatures, "set_model_link")
@patch.object(CLIPFeatures, "set_model_type")
@patch.object(CLIPFeatures, "set_precision")
@patch.object(CLIPFeatures, "set_device")
def test_clip_features_flow(
    mock_set_device,
    mock_set_precision,
    mock_set_model_type,
    mock_set_model_link,
    mock_extract,
    mock_cleanup,
    dummy_encoder,
    dummy_download,
    dummy_image_file,
):
    """
    Verify that the feature‑extraction pipeline calls the expected
    """
    # run the three blocks (copy‑paste the original snippet here)
    # block 1
    from tvln.main import main

    tensor_stack = main()
    # assertions
    assert mock_set_device.call_count == 3
    assert mock_set_precision.call_count == 3
    assert mock_set_model_link.call_count == 2
    assert mock_set_model_type.call_count == 1
    assert mock_extract.call_count == 3
    assert mock_cleanup.call_count == 3
    for name, tensors in tensor_stack.items():
        if name != "F1 VAE":
            assert isinstance(tensors[1], MagicMock)  # extraction was triggered
