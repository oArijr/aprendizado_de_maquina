import scipy.io as scipy
import Utils
import numpy as np

mat = scipy.loadmat('grupoDados1.mat')

grupoTest = mat['grupoTest']
grupoTrain = mat['grupoTrain']
testRots = mat['testRots']
trainRots = mat['trainRots']

print("Para k=1")
accuracy = Utils.accuracy(grupoTrain, trainRots, grupoTest, testRots, 1)
print(f"Acurácia: {accuracy:.0f}%\n")

print("Para k=10")
accuracy = Utils.accuracy(grupoTrain, trainRots, grupoTest, testRots, 10)
print(f"Acurácia: {accuracy:.0f}%\n")

Utils.visualizaPontos(grupoTrain, trainRots, 1, 2)
Utils.visualizaPontos(grupoTest, testRots, 1, 2)

# Q1.1. Qual é a acurácia máxima que você consegue da classificação?
acuracia_maxima, k = Utils.acuracia_maxima(grupoTrain, trainRots, grupoTest, testRots)

print(f"Maior acurácia com todas as características: {acuracia_maxima}%; K: {k}")
# Q1.2. É necessário ter todas as características (atributos) para obter a acurácia máxima para esta classificação?
for i in range(4):
    grupoTrain_coluna = np.delete(grupoTrain, i, 1)
    grupoTest_coluna = np.delete(grupoTest, i, 1)

    acuracia_maxima, k = Utils.acuracia_maxima(grupoTrain_coluna, trainRots, grupoTest_coluna, testRots)

    print(f"\nRemovendo coluna: {i} \n Maior acurácia: {acuracia_maxima}; K = {k}")

combinacoes_duplas = [[0, 1],[0, 2],[0, 3],[1, 2],[1, 3],[2, 3]]

for i in range(len(combinacoes_duplas)):
    grupoTrain_coluna = np.delete(grupoTrain, combinacoes_duplas[i], 1)
    grupoTest_coluna = np.delete(grupoTest, combinacoes_duplas[i], 1)

    acuracia_maxima, k = Utils.acuracia_maxima(grupoTrain_coluna, trainRots, grupoTest_coluna, testRots)

    print(f"\nRemovendo colunas: {combinacoes_duplas[i]} \n Maior acurácia: {acuracia_maxima}; K = {k}")


for i in range(4):
    grupoTrain_coluna = grupoTrain[:, i:i+1]
    grupoTest_coluna = grupoTest[:, i:i+1]

    acuracia_maxima, k = Utils.acuracia_maxima(grupoTrain_coluna, trainRots, grupoTest_coluna, testRots)

    print(f"\nMantendo apenas a coluna: {i} \n Maior acurácia: {acuracia_maxima}; K = {k}")


# Não, com duas colunas já é possível alcançar 98% de acurácia.
#   Com as combinações de colunas (2 "Comprimento da pétala" e 3 "Largura da pétala" e K = 4)
# e (0 "Comprimento da sépala" e 2 "Comprimento da pétala" e K = 3).
#   Vale ressaltar que chegamos a atingir 96% de acurácia
# utilizando apenas a coluna 2 (K = 5) e utilizando apenas a coluna 3 (K = 4).