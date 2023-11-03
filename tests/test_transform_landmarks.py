# file: tests/test_transform_landmarks.py
import numpy as np
from mediapipe_face_align.mediapipe_face_align import transform_landmarks


def test_transform_landmarks():
    # Create a fake landmarks and a transformation matrix
    fake_landmarks = [(10, 10), (90, 90)]
    M = np.array([[1, 0, 10], [0, 1, 20]])

    # Call the function with fake data
    transformed = transform_landmarks(fake_landmarks, M)

    # Check that the landmarks are correctly transformed
    assert transformed == [(20, 30), (100, 110)]
