import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="house_price",
    version="0.3",
    author="Pragathieshwaran",
    author_email="pragath.thirumur@tigeranalytics.com ",
    description="A package for house price prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pragathiesh/mle-training",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)