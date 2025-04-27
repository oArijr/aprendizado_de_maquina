import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter


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
            # % Calcule a distância entre o exemplo de teste e os dados de treinamento
            distancias[j] = dist(dadosTrain[j], dadosTeste[i])

        dist_ordenada = dict(sorted(distancias.items(), key=lambda item: item[1]))

        mais_proximos = list(dist_ordenada.keys())[:k]
        mais_proximos_rotulados = [rotuloTrain[mp][0] for mp in mais_proximos]

        moda = stats.mode(mais_proximos_rotulados)
        rotulo_previsto = moda.mode

        if k % 2 == 0:
            if verificar_empate(mais_proximos_rotulados):
                mais_proximos = list(dist_ordenada.keys())[:k-1]
                mais_proximos_rotulados = [rotuloTrain[mp][0] for mp in mais_proximos]
                moda = stats.mode(mais_proximos_rotulados)
                rotulo_previsto = moda.mode

        # print(f"Moda teste {i}: ", rotulo_previsto)
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
        if (rotulos[idx] == rotulo):
            ret.append(dados[idx][indice])

    return ret



def visualizaPontos(dados, rotulos, d1, d2):
    fig, ax = plt.subplots()

    cores = [
        'red', 'blue', 'green', 'orange', 'purple',
        'cyan', 'magenta', 'yellow', 'black', 'brown',
        'pink', 'gray', 'lime'
    ]

    markers = [
        '^', '+', '.', 'o', 's',
        'p', '*', 'h', 'D', 'v',
        '>', '<', 'x'
    ]

    for i in range(1, len(set([valor[0] for valor in rotulos])) + 1):
        ax.scatter(getDadosRotulo(dados, rotulos, i, d1), getDadosRotulo(dados, rotulos, i, d2), c=cores[i],
                   marker=markers[i])

    plt.show()

def normalizacao(dados):
    novos_dados = []
    for i in range(len(dados)):
        linha_nova = []
        for j in range(len(dados[i])):
            valor_atual = dados[i][j]
            max_value = np.max(dados[:, j])
            min_value = np.min(dados[:, j])
            res = (valor_atual - min_value) / (max_value - min_value)

            linha_nova.append(res)
        novos_dados.append(linha_nova)

    return novos_dados


def acuracia_maxima(dados_train, rotulos_train, dados_teste, rotulos_teste):
    acuracia_maxima = 0
    k = 0
    for i in range(1, 50):
        acuracia_atual = accuracy(dados_train, rotulos_train, dados_teste, rotulos_teste, i)
        if acuracia_maxima < acuracia_atual:
            acuracia_maxima = acuracia_atual
            k = i

    return acuracia_maxima, k




def verificar_empate(vizinhos_rotulados):
    contador = Counter(vizinhos_rotulados)
    max_contagem = max(contador.values())
    classes_empatadas = [classe for classe, cont in contador.items() if cont == max_contagem]

    if len(classes_empatadas) > 1:
        return True
    else:
        return False