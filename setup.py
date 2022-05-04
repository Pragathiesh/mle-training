from setuptools import setup

setup(
    name='house_price',
    version='0.3',
    author='Pragathieshwaran',
    author_email='pragath.thirumur@tigeranalytics.com',
    packages=['src'],
    description="Package for assignment 4.1",
    url="https://github.com/Pragathiesh/mle-training",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", ],
    python_requires='>=3.6',
)
