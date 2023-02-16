import operator
import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.manifold import TSNE

# Macros.
maxIter = 20
clustersNumber = 10
dataFile = "data.txt"
centroidsFileOne = "c1.txt"
centroidsFileOne = "c2.txt"
normUtilized = 2  # change to 1 for L1 loss


def main():
    # Configuration of settings in Spark
    configuration = SparkConf().setMaster("local").setAppName("kmeans")
    sc = SparkContext(conf=configuration)

    # Load data, we need this to access data each iteration
    data = sc.textFile(dataFile).map(
        lambda line: np.array([float(x) for x in line.split(' ')])
    ).cache()

    # Load first file of centroids, you can split into a list of np arrays
    centroidsFirst = sc.textFile(centroidsFileOne).map(
        lambda line: np.array([float(x) for x in line.split(' ')])
    ).collect()

    # Load second file of centroids, you can split into a list of np arrays
    centroidsSecond = sc.textFile(centroidsFileTwo).map(
        lambda line: np.array([float(x) for x in line.split(' ')])
    ).collect()

    print("Running k-means clustering using firsr centroids...")
    resultOne, centroidsFirst, costOne = kmeans(data=data, centroids=centroidsFirst,
                                       norm=normUtilized)

    print("Runnning k-means clustering using second centroids...")
    resultTwo, centroidsSecond, costTwo = kmeans(data=data, centroids=centroidsSecond,
                                       norm=normUtilized)

    print("Plot loss function based on norm")
    plotingLossFunction(costOne, costTwo, "picture/loss-l%d.jpg" % normUtilized)

    


if __name__ == "__main__":
    main()
