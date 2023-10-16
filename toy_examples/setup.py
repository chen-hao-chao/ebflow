# Code from https://github.com/akandykeller/SelfNormalizingFlows
from setuptools import setup

setup(
    name='EBFlow',
    version='0.0.1',
    description="EBFlow",
    author="Chen-Hao Chao",
    author_email='lance_chao@gapp.nthu.edu.tw',
    packages=[
        'ebflow'
    ],
    entry_points={
        'console_scripts': [
            'ebflow=ebflow.cli:main',
        ]
    },
    python_requires='>=3.6',
)