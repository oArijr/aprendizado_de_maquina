import scipy.io as scipy
import numpy as np
import Utils


mat = scipy.loadmat('grupoDados2.mat')
grupoTest = mat['grupoTest']
grupoTrain = mat['grupoTrain']
testRots = mat['testRots']
trainRots = mat['trainRots']

min_train, max_train = Utils.normalizacao_fit(grupoTrain)

grupoTrainNormalizado = Utils.normalizacao(grupoTrain, min_train, max_train)
grupoTestNormalizado = Utils.normalizacao(grupoTest, min_train, max_train)

Utils.visualizaPontos(grupoTrain, trainRots, 11, 12)
Utils.visualizaPontos(grupoTest, testRots, 11, 12)

#Q2.1: Aplique seu kNN a este problema. Qual é a sua acurácia de classificação?
acuracia_maxima, k = Utils.acuracia_maxima(grupoTrain, trainRots, grupoTest, testRots)
print(f"Maior acurácia com todas as características: {acuracia_maxima:.0f}%; K: {k}")

#Q2.2: A acurácia pode ser igual a 98% com o kNN. Descubra por que o resultado atual é muito menor.
# Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne 98% e explique o que você fez e por quê.

acuracia_maxima, k = Utils.acuracia_maxima(grupoTrainNormalizado, trainRots, grupoTestNormalizado, testRots)
print(f"Maior acurácia com todas as características normalizadas: {acuracia_maxima:.0f}%; K: {k}")


