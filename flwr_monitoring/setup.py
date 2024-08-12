from setuptools import setup, find_packages

setup(
    name='flwr_monitoring',
    version='0.1.0',
    description='Federated Learning Monitoring Library with ',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Kandola',
    author_email='your.email@example.com',
    url='https://github.com/kandola-network/KanFL',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'flwr', 'prometheus_client', 'psutil', 'gputil',  # Add other dependencies
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',
)
