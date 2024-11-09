import shutil
import subprocess

from setuptools import find_packages, setup
from setuptools.command.install import install


class SetupCommand(install):
    def run(self):
        print("Setting up environment")
        install.run(self)
        print("Installing pre-commit hooks")
        subprocess.run(["pre-commit", "install"])
        print("Cleaning up")
        shutil.rmtree("adding_classifier.egg-info")


setup(
    name="adding-classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "codetiming",
        "comet_ml",
        "matplotlib",
        "numpy",
        "pydantic",
        "scikit_learn",
        "scipy",
        "tqdm",
        "yaspin",
        "black",
        "isort",
        "pre-commit",
        "mypy",
        "pytest",
    ],
    cmdclass={
        "setup": SetupCommand,
    },
)
