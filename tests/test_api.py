import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server


class DummyClassifier:
    def predict(self, image_array):
        # Always returns a deterministic distribution for testing
        return np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)


def test_predict_endpoint_returns_probabilities(tmp_path):
    server._classifier = DummyClassifier()
    client = TestClient(server.app)

    sample = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.circle(sample, (112, 112), 60, (180, 180, 180), -1)
    cv2.line(sample, (40, 0), (40, 224), (90, 90, 90), 3)
    success, buf = cv2.imencode(".png", sample)
    assert success

    response = client.post(
        "/predict",
        files={"file": ("sample.png", buf.tobytes(), "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == server.CLASSES[-1]
    assert data["confidence"] == pytest.approx(0.4, rel=1e-3)
    assert set(data["probabilities"].keys()) == set(server.CLASSES)


def test_rejects_invalid_file_type():
    server._classifier = DummyClassifier()
    client = TestClient(server.app)

    response = client.post(
        "/predict",
        files={"file": ("sample.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 400


def test_rejects_non_xray_colored_image():
    server._classifier = DummyClassifier()
    client = TestClient(server.app)

    # Create a colorful pie-chart-like image
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (112, 224), (255, 0, 0), -1)
    cv2.rectangle(img, (112, 0), (224, 224), (0, 255, 0), -1)
    success, buf = cv2.imencode(".png", img)
    assert success

    response = client.post(
        "/predict",
        files={"file": ("chart.png", buf.tobytes(), "image/png")},
    )

    assert response.status_code == 400
