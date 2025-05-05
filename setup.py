from setuptools import setup, find_packages, find_namespace_packages

def read_requirements():
    with open('requirements.txt', 'r') as req_file:
        return [line.strip() for line in req_file if line.strip() and not line.startswith('#')]

setup(
    name="ICESEE",  # Your package name
    version="0.1.8",  # Initial version
    packages=find_namespace_packages(include=["ICESEE.*"]),  # Include all sub-packages
    nstall_requires=read_requirements(),
    author="Brian Kyanjo",
    author_email="briankyanjo@u.boisestate.edu",
    description="A state-of-the-art data assimilation software package for coupling ice sheet models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/KYANJO/ICESEE/tree/main",  # GitHub repo URL
    # packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)