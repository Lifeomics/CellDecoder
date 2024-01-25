from setuptools import setup, find_packages

__version__ = "0.0.1"
URL = None
install_requires = [
    "prettytable",
    "tensorboard",
    "sklearn",
    "tabulate",
    "matplotlib",
    "jupyter",
    "jupyterlab",
    "dill",
    "scanpy",
]

setup(
    name="celldecoder",
    version=__version__,
    url=URL,
    python_requires=">=3.8",
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)


# pip install torch==1.10
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
# pip install dgl-cu102 -f https://data.dgl.ai/wheels/repo.html
# pip install --upgrade "protobuf<=3.20.1"

# pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
