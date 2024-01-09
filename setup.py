from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="movement_mixer",
    version="0.0.1",
    author="Mohamed Diomande",
    author_email="gdiomande7907@gmail.com",
    description="Chain Tree is a library for generating conversational data.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/diomandeee/movement_mixer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    python_requires=">=3.6",
    install_requires=requirements,
)


