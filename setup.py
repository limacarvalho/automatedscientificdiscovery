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
    version="0.1",
    author='Arthur',
    author_email='arthur@unknown.eu',
    description="Routines for the ASD project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=install_requirements,
    package_dir={"": "src"},
)
