 
# mediapipe_face_align

`mediapipe_face_align` is a Python package for aligning faces in images using the MediaPipe Face Mesh model and transforming facial landmarks.

## Installation

You can install the package using `pip`:

```bash
pip install mediapipe_face_align
```

## Usage

Import the package and use the `process` function to align faces and get transformed landmarks.

```python
import mediapipe_face_align

# Load an image
image_path = "sample.png"

# Process the image and get the aligned face and landmarks
aligned_face, transformed_landmarks = mediapipe_face_align.process(image_path)

# Display the aligned face or use the data as needed
# You can use OpenCV to display the image:
# cv2.imshow('Aligned Face', aligned_face)
```

## Example

Here's an example of how to use the `mediapipe_face_align` package to align faces in an image:

```python
import cv2
import mediapipe_face_align

# Load an image
image_path = "sample.png"

# Process the image and get the aligned face and landmarks
aligned_face, transformed_landmarks = mediapipe_face_align.process(image_path)

# Display the aligned face
cv2.imshow('Aligned Face', aligned_face)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Features

- Aligns faces in images using facial landmarks.
- Provides transformed landmarks for further analysis or processing.
- Easy integration with OpenCV and other image processing libraries.

## Documentation

For more information and detailed documentation, please visit the [GitHub repository](https://github.com/SutirthaChakraborty/mediapipe_face_align).

## License

This package is distributed under the MIT License. See the [LICENSE](https://github.com/SutirthaChakraborty/mediapipe_face_align/blob/main/LICENSE) file for details.
 