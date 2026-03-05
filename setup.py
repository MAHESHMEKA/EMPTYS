from setuptools import setup, find_packages

setup(
    name='emptys',
    version='1.0',
    description='It is a data preprocessing library',
    author='Mahesh',
    author_email='maheshaim82@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'seaborn',
        'matplotlib',
        'datetime',
        'copy',
        'os'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
