# file: tests/test_process.py
import cv2
import pytest
from unittest.mock import Mock, patch
from mediapipe_face_align.mediapipe_face_align import process

# This will require sample image files in tests/resources/

@pytest.fixture
def mock_cv2():
    with patch('mediapipe_face_align.mediapipe_face_align.cv2') as mock:
        yield mock

@pytest.fixture
def mock_face_mesh():
    with patch('mediapipe_face_align.mediapipe_face_align.mp_face_mesh') as mock:
        yield mock

def test_process_with_face(mock_cv2, mock_face_mesh):
    # Setup fake return values for cv2 and face_mesh dependencies
    mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_face_mesh.FaceMesh.return_value.process.return_value = Mock()

    # Fake landmarks
    fake_landmarks = Mock()
    fake_landmarks.landmark = [Mock(x=0.1, y=0.1), Mock(x=0.9, y=0.9)]

    # Setup the mock to return the fake landmarks
    mock_face_mesh.FaceMesh.return_value.process.return_value.multi_face_landmarks = [fake_landmarks]

    # Call the function
    aligned_face, transformed_keypoints = process('tests/resources/test_face.jpg')

    # Assert the results
    assert aligned_face is not None
    assert transformed_keypoints is not None

def test_process_without_face(mock_cv2, mock_face_mesh):
    # Setup fake return values for cv2 and face_mesh dependencies
    mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_face_mesh.FaceMesh.return_value.process.return_value = Mock(multi_face_landmarks=None)

    # Call the function with an image that has no face
    aligned_face, transformed_keypoints = process('tests/resources/test_no_face.jpg')

    # Assert that no face was found and the aligned_face is None
    assert aligned_face is None
    assert transformed_keypoints is None
