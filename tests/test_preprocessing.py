import cv2
import numpy as np

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing import ChestXRayPreprocessor


def test_preprocess_image_shapes_and_range(tmp_path):
    image_path = tmp_path / "sample.png"
    synthetic = np.full((300, 300), 120, dtype=np.uint8)
    cv2.imwrite(str(image_path), synthetic)

    preprocessor = ChestXRayPreprocessor()
    processed = preprocessor.preprocess_image(str(image_path))

    assert processed.shape == (224, 224, 3)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
