# README

This document lists all the instructions one needs to run this tool.

First of all, you will need Python 3.6+ in order to interpret this program. 
Download Python from [here](https://www.python.org/downloads/).
In addition, one needs a host of libraries to run this program. The following is the list of all libraries needed:
* csv
* datetime
* sys
* os
* matplotlib
* pandas
* statsmodels
* sklearn
* numpy
* time

Most of these libraries are found in the standard installation of Python. The rest can be 
installed using the `pip` tool by using the commands `pip install [library_name]`. If `pip` is not installed, the instructions to install are [here](https://pip.pypa.io/en/stable/installing/).

After having everything installed, you can run the tool by using the following command in terminal/command prompt,
`python main.py "path/to/trainingFile.csv" "path/to/testFile.csv" output.csv`. The first argument is the training file while the second is the test file. Last arugment is the name you would like the output file to have. All of these files must be in csv format.
