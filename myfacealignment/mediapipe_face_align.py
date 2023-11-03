import cv2
import mediapipe as mp
import numpy as np


# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


# Define a function to align the face using eye landmarks.
def align_face(
    image: np.ndarray,
    landmarks: list,
    desired_left_eye: tuple = (0.35, 0.35),
    desired_face_width: int = 256,
    desired_face_height: int = None,
) -> tuple:
    """
    Aligns a face within an image using eye landmarks.

    Args:
        image (np.ndarray): The input image as a NumPy array (BGR format).
        landmarks (list): List of landmark points as tuples (x, y).
        desired_left_eye (tuple): Desired position of the left eye in the aligned face.
        desired_face_width (int): Desired width of the aligned face.
        desired_face_height (int): Desired height of the aligned face (default is None, will be equal to desired_face_width).

    Returns:
        tuple: A tuple containing the aligned face image (np.ndarray) and the transformation matrix (M).
    """

    if desired_face_height is None:
        desired_face_height = desired_face_width

    # The indices for the left and right eye corners.
    left_eye_idx = 130
    right_eye_idx = 359

    # Extract the left and right eye (x, y) coordinates.
    left_eye_center = landmarks[left_eye_idx]
    right_eye_center = landmarks[right_eye_idx]

    # Compute the angle between the eye centroids.
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Calculate the desired right eye x-coordinate based on the desired x-coordinate of the left eye.
    desired_right_eye_x = 1.0 - desired_left_eye[0]

    # Determine the scale of the new resulting image by taking the ratio of the distance 
    # between eyes in the current image to the ratio of distance in the desired image.
    dist = np.sqrt((dX**2) + (dY**2))
    desired_dist = desired_right_eye_x - desired_left_eye[0]
    desired_dist *= desired_face_width
    scale = desired_dist / dist

    # Compute center (x, y)-coordinates between the two eyes in the input image.
    eyes_center = (
        (left_eye_center[0] + right_eye_center[0]) // 2,
        (left_eye_center[1] + right_eye_center[1]) // 2,
    )

    # Grab the rotation matrix for rotating and scaling the face.
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    # Update the translation component of the matrix.
    tX = desired_face_width * 0.5
    tY = desired_face_height * desired_left_eye[1]
    M[0, 2] += tX - eyes_center[0]
    M[1, 2] += tY - eyes_center[1]

    # Apply the affine transformation.
    (w, h) = (desired_face_width, desired_face_height)
    output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    # Return the aligned face and the transformation matrix.
    return output, M


# Function to transform landmarks using the same transformation as the face alignment


def transform_landmarks(landmarks: list, M: np.ndarray) -> list:
    """
    Transforms a list of landmarks using a given transformation matrix.

    Args:
        landmarks (list): List of landmark points as tuples (x, y).
        M (np.ndarray): Transformation matrix.

    Returns:
        list: Transformed landmark points as tuples (x, y).
    """
    transformed_landmarks = []
    for landmark in landmarks:
        # Apply the transformation matrix to each landmark point
        x, y = landmark
        transformed_point = np.dot(M, np.array([x, y, 1]))
        transformed_landmarks.append(
            (int(transformed_point[0]), int(transformed_point[1]))
        )
    return transformed_landmarks


def process(image_path: str) -> tuple:
    """
    Processes an image to detect and align a face.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        tuple: A tuple containing the aligned face image (np.ndarray) and the transformed keypoints (list of tuples).
    """
    # Load the image from the file
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and detect the face landmarks.
    results = face_mesh.process(rgb_frame)

    transformed_frame = None
    transformed_keypoints = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert landmarks to a list of tuples (x, y).
            points = [
                (int(p.x * img.shape[1]), int(p.y * img.shape[0]))
                for p in face_landmarks.landmark
            ]

            # Align the face using the landmarks.
            aligned_face, M = align_face(img, points)

            # Transform the original landmarks to fit the aligned face
            transformed_landmarks = transform_landmarks(points, M)

            # # Draw the face mesh on the aligned face using the transformed landmarks
            # for landmark in transformed_landmarks:
            #     cv2.circle(aligned_face, landmark, 1, (0, 255, 0), -1)

            # Update the result variables
            transformed_frame = aligned_face
            transformed_keypoints = transformed_landmarks

    return transformed_frame, transformed_keypoints


# Example usage:
# aligned_face, transformed_keypoints = process("input_image.jpg")
