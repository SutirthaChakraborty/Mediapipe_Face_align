# import pytest
# import numpy as np
# import cv2
# from mediapipe_face_align import process

# # Mock an image with a face to test
# @pytest.fixture
# def mock_image_with_face():
#     # Here we create a random image that would represent an image with a face
#     # In practice, you might want to load an actual image with known outcomes
#     return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# # Test the process function with an image containing a face
# def test_process_with_face(mock_image_with_face):
#     aligned_face, transformed_keypoints = process(mock_image_with_face)
    
#     # Write assertions to check if the aligned_face and transformed_keypoints
#     # are as expected
#     assert aligned_face is not None, "Aligned face should not be None"
#     assert transformed_keypoints is not None, "Transformed keypoints should not be None"
#     assert isinstance(aligned_face, np.ndarray), "Aligned face should be a numpy array"
#     assert isinstance(transformed_keypoints, list), "Transformed keypoints should be a list"
#     assert len(transformed_keypoints) > 0, "Transformed keypoints should not be empty"

# # Test the process function with an image without a face
# def test_process_without_face():
#     # Here you create an image that does not have a face
#     non_face_image = np.zeros((480, 640, 3), dtype=np.uint8)
#     aligned_face, transformed_keypoints = process(non_face_image)
    
#     # Write assertions to check the behavior when no face is present in the image
#     assert aligned_face is None, "Aligned face should be None for non-face images"
#     assert transformed_keypoints is None, "Transformed keypoints should be None for non-face images"

# # # Run the tests
# # if __name__ == "__main__":
# #     pytest.main()
