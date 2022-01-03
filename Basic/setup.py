import setuptools

with open('README.md', encoding = 'utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name = 'ML_production',
    version = 1.0,
    description = 'Machine Learning production code skeleton',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = 'Nor Raymond',
    author_email = 'norfazlinahamdan@gmail.com',
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires = '>=3.8, <4',
    install_requires = ['pytest>=0.23.0'],
    extras_require = {'dev': ['pytest'], }, 
)