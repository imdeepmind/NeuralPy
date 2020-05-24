from setuptools import setup

long_description = """
NeuralPy

A Keras like deep learning library works on top of PyTorch

Check the docs for more info: https://neuralpy.imdeepmind.com/
"""

setup(
	name="neuralpy-torch",
	version="0.0.1",
	description="A Keras like deep learning library works on top of PyTorch",
	long_description=long_description,
	url="https://github.com/imdeepmind/NeuralPy",
	author="Abhishek Chatterjee",
	author_email="abhishek.chatterjee97@protonmail.com",
	license="MIT",
	classifiers=[
		"Development Status :: 1 - Planning",
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
		"neuralpy.optimizer",
		"neuralpy.regulariziers"
	],
	include_package_data=True
)