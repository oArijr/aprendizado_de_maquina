import scipy.io as scipy
import numpy as np
import Utils


mat = scipy.loadmat('grupoDados3.mat')
grupoTest = mat['grupoTest']
grupoTrain = mat['grupoTrain']
testRots = mat['testRots']
trainRots = mat['trainRots']

Utils.visualizaPontos(grupoTrain, trainRots, 0, 1)
Utils.visualizaPontos(grupoTest, testRots, 0, 1)

# Q3.1: Aplique o kNN ao problema usando k = 1. Qual é a acurácia na classificação?
knn1 = Utils.accuracy(grupoTrain, trainRots, grupoTest, testRots, 1)
print(f"Acurácia para K=1 é de {knn1:.0f}%")


acuracia_maxima, k = Utils.acuracia_maxima(grupoTrain, trainRots, grupoTest, testRots)
print(f"Maior acurácia com todas as características: {acuracia_maxima:.0f}%; K: {k}")

# Q3.2: A acurácia pode ser igual a 92% com o kNN. Descubra por que o resultado atual é muito menor. Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne 92% e explique o que você fez e por quê.

# O K1 nesse caso é muito ruim pois há vários classes entre as outras classes.
# O que dá o melhor resultado nesse cenário é utilizar um K maior para considerar mais vizinhos ao classificar o obj.