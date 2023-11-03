# file: tests/test_align_face.py
import numpy as np
from mediapipe_face_align.mediapipe_face_align import align_face


def test_align_face():
    # Create a fake image and fake landmarks
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Place fake landmarks at approximate positions where eyes might be located
    fake_landmarks = [(30, 30), (70, 30)] * 360  # Duplicating to fill in other indices, but keeping eyes constant.
    fake_landmarks[130] = (30, 30)  # Left eye
    fake_landmarks[359] = (70, 30)  # Right eye

    # Call the function with fake data
    aligned_image, M = align_face(fake_image, fake_landmarks)

    # Check the outputs
    assert aligned_image is not None
    assert M is not None
    assert aligned_image.shape == (256, 256, 3)  # As per the default desired dimensions
