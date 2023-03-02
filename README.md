#Submarine Structure Detection

This repository aims to implement two methods for detecting submarine structures: line detection and k-means clustering. These methods can be used to identify and analyze various underwater features, such as pipelines, cables, and geological formations.
#Line Detection Method

The line detection method involves analyzing sonar or acoustic images to identify straight lines that may correspond to submarine structures. This method utilizes various image processing techniques, such as edge detection and Hough transforms, to identify and extract lines from the image data.
#K-Means Clustering Method

The k-means clustering method involves grouping similar data points into clusters based on their features or characteristics. In the context of submarine structure detection, this method can be used to group together similar patterns or structures within the sonar or acoustic image data.
#Evaluation

To evaluate the performance of each algorithm with the datasets used in this repository, simply run the command ./evaluate in the directory where the correspondent python scripts are located. The evaluation script will output the results of the analysis, including the accuracy and precision of each algorithm.


#Requirements

The following packages are required to run the algorithms and the evaluation script:

    numpy
    matplotlib
    scikit-learn
    OpenCV

These packages can be installed using pip or any other package manager.
