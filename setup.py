"""
Setup script for the Behavioral Phishing Detection package.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='behavioral-phishing-detection',
    version='0.1.0',
    author='Elijah Adeyeye',
    author_email='eElijahadeyeye@gmail.com',
    description='An AI model that analyzes linguistic patterns to detect phishing attempts',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/handsomemedia1/Behavioral-Phishing-Detection',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Security',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'phishing-detect=src.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)