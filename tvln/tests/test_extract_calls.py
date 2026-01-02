mport builtins
import pytest
from unittest.mock import patch, MagicMock

from tefln.main import CLIPFeatures, cleanup, ModelLink, ModelType

@pytest.fixture
def dummy_gather():
    return {"text": ["a"], "image": ["b"]}

@patch("tefln.main.cleanup")
@patch.object(CLIPFeatures, "extract", return_value=MagicMock(name="tensor"))
@patch.object(CLIPFeatures, "set_model_link")
@patch.object(CLIPFeatures, "set_model_type")
@patch.object(CLIPFeatures, "set_precision")
@patch.object(CLIPFeatures, "set_device")
def test_clip_features_flow(
    mock_dev, mock_prec, mock_type, mock_link, mock_extract, mock_cleanup, dummy_gather
):
    # run the three blocks (copy‑paste the original snippet here)
    # block 1
    f = CLIPFeatures()
    f.set_device("cpu")
    f.set_precision("fp16")
    f.set_model_link(ModelLink.VIT_L_14_LAION2B_S32B_B82K)
    t1 = f.extract(dummy_gather)
    cleanup(model=f, device="cpu")

    # block 2
    f = CLIPFeatures()
    f.set_device("cpu")
    f.set_precision("fp16")
    f.set_model_type(ModelType.VIT_L_14_LAION400M_E32)
    t2 = f.extract(dummy_gather, last_layer=True)
    cleanup(model=f, device="cpu")

    # block 3
    f = CLIPFeatures()
    f.set_device("cpu")
    f.set_precision("fp16")
    f.set_model_link(ModelLink.VIT_BIGG_14_LAION2B_S39B_B160K)
    t3 = f.extract(dummy_gather)
    cleanup(model=f, device="cpu")

    # assertions
    assert mock_dev.call_count == 3
    assert mock_prec.call_count == 3
    assert mock_link.call_count == 2
    assert mock_type.call_count == 1
    assert mock_extract.call_count == 3
    assert mock_cleanup.call_count == 3
    for t in (t1, t2, t3):
        assert isinstance(t, MagicMock)   # extraction was triggered