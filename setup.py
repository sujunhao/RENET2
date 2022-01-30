import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

#def _post_install(setup):
#    def _post_actions():
#        print('asd')
#    _post_actions()
#    return setup

#setuptools.setup = _post_install(

setuptools.setup(
    name="renet2", # Replace with your own username
    version="1.3",
    author="Su Junhao",
    author_email="jhsu@cs.hku.hk",
    description="High-Performance Full-text Gene-Disease Relation Extraction with Iterative Training Data Expansion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sujunhao/RENET2",
    project_urls={
        "Bug Tracker":  "https://github.com/sujunhao/RENET2",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={
        # If any package contains *.txt files, include them:
        "renet2": ["resource/*.txt"],
        "renet2": ["utils/word_index"],
        "renet2": ["tools/geniass-1.00.tar.gz"],
    },
    #zip_safe=False,
    entry_points={  # Optional
        'console_scripts': [
            'renet2=renet2.renet2:main',
        ],
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)


