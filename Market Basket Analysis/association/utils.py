import pandas as pd
from itertools import combinations

# Função para gerar as duplas e contar ocorrências
def combinacoes_duplas(base_mercado):
    duplas = {}

    for index, row in base_mercado.iterrows():
        itens = [item for item in row if pd.notna(item)]  # Remove NaN
        for dupla in combinations(itens, 2):
            dupla_ordenada = tuple(sorted(dupla))
            duplas[dupla_ordenada] = duplas.get(dupla_ordenada, 0) + 1

    return duplas


def combinacoes_trios(base_mercado):
    trios = {}
    for index, row in base_mercado.iterrows():
        itens = [item for item in row if pd.notna(item)]
        for trio in combinations(itens, 3):
            trio_ordenada = tuple(sorted(trio))
            trios[trio_ordenada] = trios.get(trio_ordenada, 0) + 1

    return trios

def exibir(combinacoes):
    for combinacao, contagem in sorted(combinacoes.items(), key=lambda x: x[1], reverse=True):
        print(f"{combinacao}: {contagem}")


def regras_calculo_duplas(duplas, suportes, fator_confianca):
    confianca = {}
    for dupla, contagem in duplas:
        chave = f"SE {dupla[0]} ENTAO {dupla[1]}"
        calculo = contagem / suportes[dupla[0]]
        if calculo > fator_confianca:
            confianca[chave] = calculo

        chave = f"SE {dupla[1]} ENTAO {dupla[0]}"
        calculo = contagem / suportes[dupla[1]]
        if calculo > fator_confianca:
            confianca[chave] = calculo

    return confianca
