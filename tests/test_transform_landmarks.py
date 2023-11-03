# file: tests/test_transform_landmarks.py
import numpy as np
from mediapipe_face_align.mediapipe_face_align import transform_landmarks

import numpy.testing as npt

def test_transform_landmarks():
    # Create a fake landmarks and a transformation matrix
    fake_landmarks = np.array([(10, 10), (90, 90)])
    M = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]])

    # Call the function with fake data
    transformed = transform_landmarks(fake_landmarks, M)

    # Check that the landmarks are correctly transformed
    expected_transformed = np.array([(20, 30), (100, 110)])
    npt.assert_array_equal(transformed, expected_transformed)
