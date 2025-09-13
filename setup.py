from setuptools import setup, find_packages

setup(
    name="nova-demo-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pytest",
        "pytest-json-report",
    ],
)
