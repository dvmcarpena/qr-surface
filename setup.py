from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="tfginfo",
    version="1.0.0",
    packages=[
        "tfginfo",
    ],
    install_requires=requirements,
    author="David Martinez Carpena",
    author_email="dvmcarpena@pm.me",
    url="https://dvmcarpena.com/",
    license="",
    description="",
    long_description=long_description
)
