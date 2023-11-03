# file: tests/test_align_face.py
import numpy as np
from mediapipe_face_align.mediapipe_face_align import align_face

def test_align_face():
    # Create a fake image and fake landmarks
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_landmarks = [(10, 10), (90, 90)]

    # Call the function with fake data
    aligned_image, M = align_face(fake_image, fake_landmarks)

    # Check the outputs
    assert aligned_image is not None
    assert M is not None
    assert aligned_image.shape == (256, 256, 3)  # As per the default desired dimensions
