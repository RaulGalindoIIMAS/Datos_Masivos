import operator
import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.manifold import TSNE

# Macros.
MAX_ITER = 20
DATA_PATH = "data.txt"
C1_PATH = "c1.txt"
C2_PATH = "c2.txt"
NORM = 2  # change to 1 for l1 loss


def main():
    # Spark settings
    configuration = SparkConf().setMaster("local").setAppName("kmeans")
    sc = SparkContext(conf=configuration)

    # Load the data, cache this since we're accessing this each iteration
    data = sc.textFile(DATA_PATH).map(
        lambda line: np.array([float(x) for x in line.split(' ')])
    ).cache()
    # Load the initial centroids c1, split into a list of np arrays
    centroidsFirst = sc.textFile(C1_PATH).map(
        lambda line: np.array([float(x) for x in line.split(' ')])
    ).collect()
    # Load the initial centroids c2, split into a list of np arrays
    centroidsSecond = sc.textFile(C2_PATH).map(
        lambda line: np.array([float(x) for x in line.split(' ')])
    ).collect()
    print("Run kmean clustering.")
    resultOne, centroidsFirst, costOne = kmeans(data=data, centroids=centroidsFirst,
                                       norm=NORM)

    print("Run kmean++ clustering.")
    resultTwo, centroidsSecond, costTwo = kmeans(data=data, centroids=centroidsSecond,
                                       norm=NORM)
    print("Plot loss.")
    plot_loss(costOne, costTwo, "picture/loss-l%d.jpg" % NORM)

    


if __name__ == "__main__":
    main()
