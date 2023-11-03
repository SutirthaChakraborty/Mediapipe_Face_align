from setuptools import setup, find_packages

setup(
    name="mediapipe_face_align",
    version="0.1",
    description="A package for face alignment",
    author="Sutirtha Chakraborty",
    author_email="sutirtha@example.com",
    url="https://github.com/SutirthaChakraborty/Mediapipe_Face_align",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "mediapipe",
        "numpy",
    ],
)
