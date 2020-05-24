from setuptools import setup

def get_readme():
	with open("README.md", "r") as f:
		return f.read()

setup(
	name="neuralpy",
	version="0.0.1",
	description=" A Keras like deep learning library works on top of PyTorch",
	long_description=get_readme(),
	long_description_content_type="text/markdown",
	url="https://github.com/imdeepmind/NeuralPy",
	author="Abhishek Chatterjee",
	author_email="abhishek.chatterjee97@protonmail.com",
	license="MIT",
	classifiers=[
		"Development Status :: Unstable",
		"Intended Audience :: Developers",
		"Intended Audience :: Education",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: MIT License",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.6",
		"Programming Language :: Python :: 3.7",
		"Topic :: Software Development :: Libraries"
		"Topic :: Software Development :: Libraries :: Python Modules"
	],
	packages=["neuralpy"],
	include_package_data=True
)