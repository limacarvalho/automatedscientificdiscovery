import setuptools
from pathlib import Path

#insert requirements in setup.py for usage and develop
root_path = Path(__file__).parent
with open(root_path / "requirements.txt") as f:
    install_requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="asd",
    version="1.0",
    author='Joao Carvalho',
    author_email='joao@limacarvalho.com',
    description="ASD project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8.10",
    install_requires=install_requirements,
    package_dir={"": "src"},
)
