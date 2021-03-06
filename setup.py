from setuptools import setup

long_description = """
NeuralPy

NeuralPy is a Keras like, deep learning library that works on top of PyTorch
written purely in Python. It is simple, easy to use library, cross-compatible with
PyTorch models, suitable for all kinds of machine learning experiments, learning,
research, etc.

Check the docs for more info: https://www.neuralpy.xyz/
"""

setup(
    name="neuralpy-torch",
    version="0.2.0",
    description="A Keras like deep learning library works on top of PyTorch",
    long_description=long_description,
    url="https://www.neuralpy.xyz/",
    author="Abhishek Chatterjee",
    author_email="abhishek.chatterjee97@protonmail.com",
    license="MIT",
    project_urls={
        "Bug Tracker": "https://github.com/imdeepmind/NeuralPy/issues",
        "Documentation": "https://www.neuralpy.xyz/",
        "Source Code": "https://github.com/imdeepmind/NeuralPy",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=[
        "neuralpy",
        "neuralpy.layers",
        "neuralpy.layers.activation_functions",
        "neuralpy.layers.convolutional",
        "neuralpy.layers.linear",
        "neuralpy.layers.normalization",
        "neuralpy.layers.other",
        "neuralpy.layers.pooling",
        "neuralpy.layers.recurrent",
        "neuralpy.layers.regularizers",
        "neuralpy.layers.sparse",
        "neuralpy.loss_functions",
        "neuralpy.models",
        "neuralpy.optimizer",
        "neuralpy.callbacks",
    ],
    install_requires=["torch"],
    extras_require={"tests": ["pytest", "pytest-cov", "flake8", "black"]},
    include_package_data=True,
)
