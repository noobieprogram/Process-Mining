# README

This document lists all the instructions one needs to run this tool.

First of all, you will need Python 3.6+ in order to interpret this program. 
Download Python from [here](https://www.python.org/downloads/).
In addition, a few libraries are also needed. The following is the list of all libraries needed:
* csv
* datetime
* sys
* warnings
* matplotlib
* pandas
* multiprocessing
* sklearn
* numpy
* time

Most of these libraries are installed by default in Python. The rest can be 
installed using the `pip` tool by using the command `pip install [library_name]`. If `pip` is not installed, the instructions to install are [here](https://pip.pypa.io/en/stable/installing/).

After having everything installed, you can run the tool by using the following command in terminal/command prompt,
`python main.py "path/to/trainingFile.csv" "path/to/testFile.csv" "output.csv"`. The first two arguments are the paths to training and test files espectively while the last argument is the name or path to the output file. All of these files must be in csv format.

After termination, an output file in the csv format will left in the directory of `main.py`. The file file will contain three addtional columns containing predictions.
