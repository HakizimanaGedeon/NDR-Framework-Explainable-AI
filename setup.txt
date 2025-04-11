from setuptools import setup, find_packages

setup(
    name='ndr',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.6.0',
        'rdflib>=5.0.0',
        'numpy>=1.20.0',
        'pandas>=1.2.0',
        'matplotlib>=3.4.0'
    ],
    description='Nomological Deductive Reasoning Framework for Explainable AI',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/ndr',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
