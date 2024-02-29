from setuptools import setup, find_packages

setup(
    name="bitnet158",  # Name of the package
    version="0.1",  # Version of the package
    description="",  # Short description
    long_description_content_type="text/markdown",  # Type of the long description
    author="Entropy Xu",  # Name of the author
    author_email="entropy.xuceyu@gmail.com",  # Email of the author
    packages=find_packages(['bitnet158']),
    classifiers=[  # Classifiers help users find your project by categorizing it.
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires='>=3.9',  # Minimum version of Python required
    install_requires=[  # List of dependencies
    ],
)
