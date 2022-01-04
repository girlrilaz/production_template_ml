import setuptools

with open('README.md', encoding = 'utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name = 'RandomForest',
    version = '0.0.1',
    description = 'Machine Learning production code skeleton',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = 'Nor Raymond',
    author_email = 'norfazlinahamdan@gmail.com',
    url="https://github.com/girlrilaz/production_template_ml",
    project_urls={
        "Bug Tracker": "https://github.com/girlrilaz/production_template_ml/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires = '>=3.8, <4',
    install_requires = ["pandas>=0.23.1",
                        "scikit-learn>=0.21.0",
                        "joblib",
                        "click",
                        "flask",
                    ],
    extras_require = {'dev': ["pep517", "pytest", "pylint", "black", "pre-commit"], 
    }, 
    entry_points={"console_scripts": ["RandomForest=RandomForest.cli:cli"]},
)