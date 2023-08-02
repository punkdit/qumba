from setuptools import setup, find_packages

setup(
    name='qumba',
    version='0.0.1',
    url='https://github.com/punkdit/qumba',
    author='Simon Burton',
    author_email='simon@arrowtheory.com',
    description='quantum stabilizer codes',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1'],
)
