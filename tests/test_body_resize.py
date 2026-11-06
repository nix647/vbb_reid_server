"""Body resize option (2026-11-06): the nightly path must preprocess like the vectors the owner's
E20 test was scored on (PIL bilinear, as in CLIP-ReID's own transforms), not cv2 INTER_LINEAR."""

import numpy as np
import pytest


def _model(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 3, 256, 128])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 3])
    node = helper.make_node("ReduceMean", ["x"], ["y"], axes=[2, 3], keepdims=0)
    model = helper.make_model(helper.make_graph([node], "g", [x], [y]), opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    p = tmp_path / "body.onnx"
    onnx.save(model, str(p))
    return p


def _crop():
    return np.random.default_rng(3).integers(0, 256, size=(301, 157, 3), dtype=np.uint8)  # BGR, not a multiple


def test_pil_resize_matches_pil_bilinear_exactly(tmp_path):
    from PIL import Image
    from vbb_reid_server.models.onnx_models import OnnxBodyEmbedder
    crop = _crop()
    e = OnnxBodyEmbedder(_model(tmp_path), mean=(127.5,) * 3, std=(127.5,) * 3, resize="pil")
    rgb = np.ascontiguousarray(crop[..., ::-1])
    ref = np.asarray(Image.fromarray(rgb).resize((128, 256), Image.Resampling.BILINEAR)).astype(np.float32)
    ref = ((ref - 127.5) / 127.5).transpose(2, 0, 1)[None]
    assert np.array_equal(e._prep([crop]), ref)


def test_cv2_default_is_unchanged_and_differs_from_pil(tmp_path):
    import cv2
    from vbb_reid_server.models.onnx_models import OnnxBodyEmbedder
    p = _model(tmp_path)
    crop = _crop()
    cv = OnnxBodyEmbedder(p)
    pil = OnnxBodyEmbedder(p, resize="pil")
    ref = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)[..., ::-1].astype(np.float32)
    ref = ((ref - cv.mean) / cv.std).transpose(2, 0, 1)[None]
    assert np.array_equal(cv._prep([crop]), ref)
    assert not np.allclose(cv._prep([crop]), pil._prep([crop]), atol=1e-3)  # the option really changes the input


def test_resize_is_part_of_the_model_id(tmp_path):
    from vbb_reid_server.models.onnx_models import OnnxBodyEmbedder
    p = _model(tmp_path)
    cv, pil = OnnxBodyEmbedder(p).model_id, OnnxBodyEmbedder(p, resize="pil").model_id
    assert cv != pil  # a cache made with one resize is never read as the other
    assert ":pil" not in cv and pil.endswith(":pil")  # existing ids (and caches) are unchanged


def test_unknown_resize_is_refused(tmp_path):
    from vbb_reid_server.models.onnx_models import OnnxBodyEmbedder
    with pytest.raises(ValueError):
        OnnxBodyEmbedder(_model(tmp_path), resize="lanczos")
