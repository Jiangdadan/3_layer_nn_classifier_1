# setup.py
from setuptools import setup, find_packages

setup(
    name="3_layer_nn_classifier",
    description="A simple CIFAR-10 classifier",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/cifar10_classifier",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "matplotlib",
        "tqdm",
        # 添加其他依赖
    ],
    entry_points={
        "console_scripts": [
            "cifar10-train=train:main",
        ],
    },
)