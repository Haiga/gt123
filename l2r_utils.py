from __future__ import division

from scipy import stats
from scipy.stats import norm
import scipy.stats as st
from subprocess import call
import numpy as np
import sys
import re
import math


def readingFile(file):
    line = None
    AntGenes = "";
    ap = []

    with open(file, 'r') as f:
        for line in f:
            m = re.search('.*\s*mean=*>(.*)\s', line)
            if m:
                ap.append(float(m.group(1)))

    return np.asarray(ap)


def getMeanRiskBaseline(mat):  ##this fuction servers to get the average prediction from features

    baseline = np.array([0.0] * mat.shape[0])
    for i in range(mat.shape[0]):
        baseline[i] = np.mean(mat[i, :])

    return baseline


def getMaxRiskBaseline(mat):  ##this fuction servers to get the feature with better prediction
    baseline = np.array([0.0] * mat.shape[0])
    for i in range(mat.shape[0]):
        baseline[i] = np.max(mat[i, :])

    return baseline


def getFullBaselineByFold(coll, l2r, fold):
    fileName = "/home/daniel/Dropbox/WorkingFiles/L2R_Baselines/" + coll + "." + l2r + ".ndcg.test.Fold" + str(fold)
    ndcgBaseline = readingFile(fileName)

    return np.asarray(ndcgBaseline)


def getFullBaseline(coll, l2r, MAX_Fold):
    ndcgBaseline = []
    for fold in range(1, MAX_Fold):
        fileName = "/home/daniel/Dropbox/WorkingFiles/L2R_Baselines/" + coll + "." + l2r + ".ndcg.test.Fold" + str(fold)
        temp = readingFile(fileName)
        ndcgBaseline = ndcgBaseline + temp.tolist()

    return np.asarray(ndcgBaseline)


def getGeoRisk(mat, alpha):
    ##### IMPORTANT
    # This function takes a matrix of number of rows as a number of queries, and the number of collumns as the number of systems.
    ##############
    numSystems = mat.shape[1]
    numQueries = mat.shape[0]
    Tj = np.array([0.0] * numQueries)
    Si = np.array([0.0] * numSystems)
    geoRisk = np.array([0.0] * numSystems)
    zRisk = np.array([0.0] * numSystems)
    mSi = np.array([0.0] * numSystems)

    for i in range(numSystems):
        Si[i] = np.sum(mat[:, i])
        mSi[i] = np.mean(mat[:, i])

    for j in range(numQueries):
        Tj[j] = np.sum(mat[j, :])

    N = np.sum(Tj)

    for i in range(numSystems):
        tempZRisk = 0
        for j in range(numQueries):
            eij = Si[i] * (Tj[j] / N)
            xij_eij = mat[j, i] - eij
            if eij != 0:
                ziq = xij_eij / math.sqrt(eij)
            else:
                ziq = 0
            if xij_eij < 0:
                ziq = (1 + alpha) * ziq
            tempZRisk = tempZRisk + ziq
        zRisk[i] = tempZRisk

    c = numQueries
    for i in range(numSystems):
        ncd = norm.cdf(zRisk[i] / c)
        geoRisk[i] = math.sqrt((Si[i] / c) * ncd)

    return geoRisk


print(getGeoRisk(np.array([[0.1, 0.2], [0.1, 0.2]]), 2.0))
