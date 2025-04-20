import math
import numpy as np
from scipy import stats


def dist(vetor_p, vetor_q):
    total = 0
    for p, q in zip(vetor_p, vetor_q):
        total += np.pow(p - q, 2)
    return math.sqrt(total)


def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
    for i in dadosTeste:
        distancias = []
        for j in dadosTrain:
            # % Calcule a distância entre o exemplo de teste e os dados de treinamento
            distancias.append(dist(dadosTrain[j], dadosTeste[i]))

        distOrdenada = sorted(distancias)

        mais_proximos = distOrdenada[:k]

        moda = stats.mode(mais_proximos)
        res = moda.mode[0]
        print(f"Moda teste {i}: ", res)
        print(f"Rótulo teste {i}: ", rotuloTrain[i])
