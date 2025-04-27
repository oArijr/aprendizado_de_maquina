import scipy.io as scipy
import numpy as np
import Utils


mat = scipy.loadmat('grupoDados4.mat')
grupoTest = mat['testSet']
grupoTrain = mat['trainSet']
testRots = mat['testLabs']
trainRots = mat['trainLabs']


# Q4.1: Aplique seu algoritmo K-NN ao problema. Qual é a sua acurácia de classificação?
acuracia_maxima, k = Utils.acuracia_maxima(grupoTrain, trainRots, grupoTest, testRots)
print(f"Maior acurácia com todas as características: {acuracia_maxima:.0f}%; K: {k}")



grupoTrainNormalizado = Utils.normalizacao(grupoTrain)
grupoTestNormalizado = Utils.normalizacao(grupoTest)
acuracia_maxima, k = Utils.acuracia_maxima(grupoTrainNormalizado, trainRots, grupoTestNormalizado, testRots)
print(f"Maior acurácia com todas as características normalizadas: {acuracia_maxima:.0f}%; K: {k}")

# Q4.2: A acurácia pode chegar a 92% com o K-NN. Descubra por que o resultado atual é muito menor. Ajuste o conjunto de dados ou o valor de k de forma que a acurácia atinja 92% e explique o que você fez e por quê. Observe que, desta vez, há mais de um problema...
for i in range(4):
    grupoTrainNormalizado_coluna = np.delete(grupoTrainNormalizado, i, 1)
    grupoTestNormalizado_coluna = np.delete(grupoTestNormalizado, i, 1)

    acuracia_maxima, k = Utils.acuracia_maxima(grupoTrainNormalizado_coluna, trainRots, grupoTestNormalizado_coluna, testRots)

    print(f"\nRemovendo coluna: {i} \n Maior acurácia: {acuracia_maxima}; K = {k}")

combinacoes_duplas = [[0, 1],[0, 2],[0, 3],[1, 2],[1, 3],[2, 3]]

for i in range(len(combinacoes_duplas)):
    grupoTrainNormalizado_coluna = grupoTrainNormalizado[:, combinacoes_duplas[i]]
    grupoTestNormalizado_coluna = grupoTestNormalizado[:, combinacoes_duplas[i]]

    acuracia_maxima, k = Utils.acuracia_maxima(grupoTrainNormalizado_coluna, trainRots, grupoTestNormalizado_coluna, testRots)

    print(f"\nMantendo as colunas: {combinacoes_duplas[i]} \n Maior acurácia: {acuracia_maxima}; K = {k}")


for i in range(4):
    grupoTrainNormalizado_coluna = grupoTrainNormalizado[:, i:i+1]
    grupoTestNormalizado_coluna = grupoTestNormalizado[:, i:i+1]

    acuracia_maxima, k = Utils.acuracia_maxima(grupoTrainNormalizado_coluna, trainRots, grupoTestNormalizado_coluna, testRots)

    print(f"\nMantendo apenas a coluna: {i} \n Maior acurácia: {acuracia_maxima}; K = {k}")