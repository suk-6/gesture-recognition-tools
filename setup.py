import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gesture_recognition_tools",
    version="1.0.0",
    author="suk-6",
    author_email="me@suk.kr",
    description="Packages for simple implementations of Gesture Recognition models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suk-6/gesture-recognition-tools.git",
    packages=setuptools.find_packages(
        include=["gesture_recognition_tools", "gesture_recognition_tools.*"]
    ),
    install_requires=["opencv-python", "openvino", "openvino-dev", "scipy"],
    python_requires=">=3.6",
    license="MIT",
)
