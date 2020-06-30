from setuptools import setup

long_description = """
NeuralPy

NeuralPy is a Keras like, deep learning library that works on top of PyTorch written purely in Python. It is simple, easy to use library, cross-compatible with PyTorch models, suitable for all kinds of machine learning experiments, learning, research, etc.

Check the docs for more info: https://neuralpy.imdeepmind.com/
"""

setup(
    name="neuralpy-torch",
    version="0.1.1",
    description="A Keras like deep learning library works on top of PyTorch",
    long_description=long_description,
    url="https://neuralpy.imdeepmind.com/",
    author="Abhishek Chatterjee",
    author_email="abhishek.chatterjee97@protonmail.com",
    license="MIT",
    project_urls={
            "Bug Tracker": "https://github.com/imdeepmind/NeuralPy/issues",
            "Documentation": "https://neuralpy.imdeepmind.com/",
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
        "Programming Language :: Python :: 3.7"
    ],
    packages=[
        "neuralpy",
        "neuralpy.activation_functions",
        "neuralpy.layers",
        "neuralpy.loss_functions",
        "neuralpy.models",
        "neuralpy.optimizer",
        "neuralpy.regulariziers"
    ],
    install_requires=["torch"],
    extras_require={
        'tests': [
            "pytest",
            "pytest-cov"
        ]
    },
    include_package_data=True
)
