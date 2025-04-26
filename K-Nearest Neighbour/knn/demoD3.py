import scipy.io as scipy
import numpy as np
import Utils


mat = scipy.loadmat('grupoDados3.mat')
grupoTest = mat['grupoTest']
grupoTrain = mat['grupoTrain']
testRots = mat['testRots']
trainRots = mat['trainRots']


# Q3.1: Aplique o kNN ao problema usando k = 1. Qual é a acurácia na classificação?
knn1 = Utils.accuracy(grupoTrain, trainRots, grupoTest, testRots, 1)
print(f"Acurácia para K=1 é de {knn1:.0f}%")

acuracia_maxima, k = Utils.acuracia_maxima(grupoTrain, trainRots, grupoTest, testRots)
print(f"Maior acurácia com todas as características: {acuracia_maxima:.0f}%; K: {k}")