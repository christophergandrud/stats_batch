from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='stats_batch',
    version='0.0.9000',    
    description='Find statistics (e.g. mean and variance) using batch updating algorithms',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/christophergandrud/stats_batch',
    project_urls={
        "Bug Tracker": "https://github.com/christophergandrud/stats_batch/issues",
    },
    author='Christopher Gandrud',
    author_email='christopher.gandrud@gmail.com',
    license='MIT',
    packages=['stats_batch'],
    install_requires=['numpy',
                      'scipy'                   
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3.5',
    ]
)