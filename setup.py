from setuptools import setup, find_packages

setup(
    name="pln-analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "torch>=2.3.0",
        "torchvision>=0.20.0",
        "scikit-learn>=1.4.2",
        "pandas>=2.2.2",
        "matplotlib>=3.8.4",
        "seaborn>=0.13.2",
        "huggingface-hub>=0.23.0",
        "tqdm>=4.66.4",
        "streamlit>=1.36.0",
    ],
)