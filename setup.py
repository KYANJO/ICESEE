from setuptools import setup, find_packages

setup(
    name="ICESEE",  # Your package name
    version="0.1.0",  # Initial version
    author="Brian Kyanjo",
    author_email="briankyanjo@u.boisestate.edu",
    description="A state-of-the-art data assimilation software package for coupling ice sheet models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/KYANJO/ICESEE/tree/main",  # GitHub repo URL
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[],  # Add dependencies here if needed
)