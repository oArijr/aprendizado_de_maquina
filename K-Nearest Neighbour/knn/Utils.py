import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def dist(vetor_p, vetor_q):
    total = 0
    for p, q in zip(vetor_p, vetor_q):
        total += np.pow(p - q, 2)
    return math.sqrt(total)


def meu_knn(dadosTrain, rotuloTrain, dadosTeste, k):
    rotulos_previstos = []

    for i in range(len(dadosTeste)):
        distancias = {}
        for j in range(len(dadosTrain)):
            distancias[j] = dist(dadosTrain[j], dadosTeste[i])

        dist_ordenada = dict(sorted(distancias.items(), key=lambda item: item[1]))

        mais_proximos = list(dist_ordenada.keys())[:k]
        mais_proximos_rotulados = [rotuloTrain[mp][0] for mp in mais_proximos]

        moda = stats.mode(mais_proximos_rotulados)
        rotulo_previsto = moda.mode
        rotulos_previstos.append(rotulo_previsto)
    return rotulos_previstos


def accuracy(dados_train, rotulos_train, dados_teste, rotulos_teste, k):
    rotulos_previstos = meu_knn(dados_train, rotulos_train, dados_teste, k)

    estaCorreto = [rotulos_teste[i][0] == rot_prev for i, rot_prev in enumerate(rotulos_previstos)]

    numCorreto = sum(estaCorreto)

    totalNum = len(rotulos_previstos)

    return (numCorreto / totalNum) * 100



def getDadosRotulo(dados, rotulos, rotulo, indice):
    ret = []
    for idx in range(0, len(dados)):
        if(rotulos[idx] == rotulo):
            ret.append(dados[idx][indice])

    return ret



def visualizaPontos(dados, rotulos, d1, d2):
    fig, ax = plt.subplots()

    ax.scatter(getDadosRotulo(dados, rotulos, 1, d1), getDadosRotulo(dados, rotulos, 1, d2), c='red' , marker='^')

    ax.scatter(getDadosRotulo(dados, rotulos, 2, d1), getDadosRotulo(dados, rotulos, 2, d2), c='blue' , marker='+')

    ax.scatter(getDadosRotulo(dados, rotulos, 3, d1), getDadosRotulo(dados, rotulos, 3, d2), c='green', marker='.')

    plt.show()

def acuracia_maxima(dados_train, rotulos_train, dados_teste, rotulos_teste):
    acuracia_maxima = 0
    k = 0
    for i in range(1, 50):
        acuracia_atual = accuracy(dados_train, rotulos_train, dados_teste, rotulos_teste, i)
        if acuracia_maxima < acuracia_atual:
            acuracia_maxima = acuracia_atual
            k = i

    return acuracia_maxima, k








