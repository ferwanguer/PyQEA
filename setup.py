from setuptools import setup, find_packages

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
    name='PyQEA',
    version='0.1.5',
    description="General use optimizer for non-convex cost\
                functions with non-linear constraints",
    author='Fernando Wanguemert',
    license_files=('LICENSE',),
    long_description=long_description,
    url='https://github.com/ferwanguer/PyQEA',
    long_description_content_type="text/markdown",
    author_email='f.wguerra@outlook.com',
    packages=find_packages(),  # same as name
    classifiers=[
                 "Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        ],
)

# test 