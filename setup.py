from setuptools import setup, find_packages

setup(
    name="transformer_from_scratch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A transformer architecture implementation from scratch",
    keywords="transformer, deep learning, NLP, attention",
    url="https://github.com/yourusername/transformer_from_scratch",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)