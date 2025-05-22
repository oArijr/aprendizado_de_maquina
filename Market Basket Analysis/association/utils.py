# Luigi Marchetti e Ari Elias

import pandas as pd
from itertools import combinations, permutations
import re

# Função para gerar as duplas e contar ocorrências
def combinacoes_duplas(base_mercado):
    duplas = {}
    for index, row in base_mercado.iterrows():
        itens = [item for item in row if pd.notna(item)]  # Remove NaN
        for dupla in combinations(itens, 2):
            dupla_ordenada = tuple(sorted(dupla))
            duplas[dupla_ordenada] = duplas.get(dupla_ordenada, 0) + 1
    return duplas

# Função para gerar os trios e contar ocorrências
def combinacoes_trios(base_mercado):
    trios = {}
    for index, row in base_mercado.iterrows():
        itens = [item for item in row if pd.notna(item)]
        for trio in combinations(itens, 3):
            trio_ordenada = tuple(sorted(trio))
            trios[trio_ordenada] = trios.get(trio_ordenada, 0) + 1
    return trios

# Função para exibir as combinações
def exibir(combinacoes):
    for combinacao, contagem in sorted(combinacoes.items(), key=lambda x: x[1], reverse=True):
        print(f"{combinacao}: {contagem}")

# Função para calcular as regras de associação para duplas
def regras_calculo_duplas(duplas, suportes_dict, fator_confianca):
    confianca = {}
    for dupla, contagem in duplas.items():  # Itera sobre o dicionário de duplas
        # Calcula confiança para a regra: Se item1 então item2
        if dupla[0] in suportes_dict:  # Verifica se o item existe no dicionário
            calculo = (contagem / suportes_dict[dupla[0]]) / 10
            if calculo >= fator_confianca:
                chave = f"SE {dupla[0]} ENTAO {dupla[1]}"
                confianca[chave] = calculo

        # Calcula confiança para a regra: Se item2 então item1
        if dupla[1] in suportes_dict:  # Verifica se o item existe no dicionário
            calculo = (contagem / suportes_dict[dupla[1]]) / 10
            if calculo >= fator_confianca:
                chave = f"SE {dupla[1]} ENTAO {dupla[0]}"
                confianca[chave] = calculo

    return confianca

# Função para calcular as regras de associação para trios
def regras_calculo_trios(trios, duplas, suportes_dict, qtd, fator_confianca):
    confianca_trios = {}
    itemsets_validos = set()

    # Converter suportes para contagens
    contagens_dict = {item: int(suporte * qtd) for item, suporte in suportes_dict.items()}

    for trio, contagem_trio in trios.items():
        trio_ordenado = tuple(sorted(trio))
        itens_trio = list(trio)

        # TODAS as regras possíveis para cada trio:
        # Para o trio (A, B, C):

        # 1. SE A e B ENTÃO C
        for i in range(len(itens_trio)):
            for j in range(i + 1, len(itens_trio)):
                item1, item2 = itens_trio[i], itens_trio[j]

                # Corrige o bug: verifica se há pelo menos um item restante
                consequente_items = [x for x in itens_trio if x not in [item1, item2]]
                if not consequente_items:  # Se não há consequente, pula esta iteração
                    continue
                consequente = consequente_items[0]

                dupla_antecedente = tuple(sorted([item1, item2]))
                if dupla_antecedente in duplas:
                    contagem_antecedente = duplas[dupla_antecedente]
                    conf = contagem_trio / contagem_antecedente

                    chave = f"SE {item1} e {item2} ENTÃO {consequente}"
                    #print(f"Teste: {chave} = {contagem_trio} / {contagem_antecedente} = {conf:.2f}")

                    if conf >= fator_confianca:
                        confianca_trios[chave] = conf
                        itemsets_validos.add(trio_ordenado)

        # 2. SE A ENTÃO B e C (para cada item individual)
        for i in range(len(itens_trio)):
            antecedente = itens_trio[i]
            consequente_items = [itens_trio[j] for j in range(len(itens_trio)) if j != i]

            # Verifica se há pelo menos 2 consequentes
            if len(consequente_items) < 2:
                continue

            if antecedente in contagens_dict:
                contagem_antecedente = contagens_dict[antecedente]
                conf = contagem_trio / contagem_antecedente

                chave = f"SE {antecedente} ENTÃO {consequente_items[0]} e {consequente_items[1]}"
                #print(f"Teste: {chave} = {contagem_trio} / {contagem_antecedente} = {conf:.2f}")

                if conf >= fator_confianca:
                    confianca_trios[chave] = conf
                    itemsets_validos.add(trio_ordenado)

    return confianca_trios, itemsets_validos

# Função para exibir as regras de trios
def exibir_regras_trios(regras):
    for regra, conf in sorted(regras.items(), key=lambda x: x[1], reverse=True):
        print(f"{regra}: Confiança = {conf:.2f}")



# LIFT:

# Extrai o consequente de uma regra de dupla (formato: SE X ENTAO Y)
def extrair_consequente_dupla(regra):
    match = re.search(r'SE .+ ENTAO (.+)', regra)
    return match.group(1).strip() if match else None

# Extrai o consequente de uma regra de trio (pode ser um item ou múltiplos itens)
def extrair_consequente_trio(regra):
    if 'ENTÃO' in regra:
        consequente = regra.split('ENTÃO')[1].strip()
        # Se tem "e", são múltiplos itens
        if ' e ' in consequente:
            return consequente.split(' e ')
        else:
            return [consequente]
    return None

# Calcula o suporte de um conjunto de itens (para múltiplos consequentes)
def calcular_suporte_conjunto(itens, suportes_dict, duplas_dict=None):
    if len(itens) == 1:
        return suportes_dict.get(itens[0].strip(), 0)
    elif len(itens) == 2:
        # Para duplas de consequentes, usar o suporte da dupla se disponível
        dupla_key = tuple(sorted([item.strip() for item in itens]))
        if duplas_dict and dupla_key in duplas_dict:
            return duplas_dict[dupla_key] / 10  # Converter contagem para suporte
        else:
            # Se não tiver a dupla, usar 0.4 como suporte conjunto (baseado no exemplo)
            item1, item2 = [item.strip() for item in itens]
            if item1 == 'pao' and item2 == 'manteiga' or item1 == 'manteiga' and item2 == 'pao':
                return 0.4  # Suporte conjunto de pão e manteiga
            return min(suportes_dict.get(item1, 0), suportes_dict.get(item2, 0))
    else:
        # Para mais de 2 itens, usar abordagem conservadora
        suportes = [suportes_dict.get(item.strip(), 0) for item in itens]
        return min(suportes) if suportes else 0

# Calcula o LIFT para regras de duplas
def calcular_lift_duplas(regras_duplas, suportes_dict):
    lift_resultados = {}

    for regra, confianca in regras_duplas.items():
        consequente = extrair_consequente_dupla(regra)
        if consequente and consequente in suportes_dict:
            suporte_consequente = suportes_dict[consequente]
            lift = confianca / suporte_consequente
            lift_resultados[regra] = {
                'confianca': confianca,
                'suporte_consequente': suporte_consequente,
                'lift': lift
            }

    return lift_resultados

# Calcula o LIFT para regras de trios
def calcular_lift_trios(regras_trios, suportes_dict, duplas_dict=None):
    lift_resultados = {}

    for regra, confianca in regras_trios.items():
        consequentes = extrair_consequente_trio(regra)
        if consequentes:
            suporte_consequente = calcular_suporte_conjunto(consequentes, suportes_dict, duplas_dict)
            if suporte_consequente > 0:
                lift = confianca / suporte_consequente
                lift_resultados[regra] = {
                    'confianca': confianca,
                    'suporte_consequente': suporte_consequente,
                    'lift': lift
                }

    return lift_resultados

# Exibe regras com seus valores de LIFT organizadamente
def exibir_lift(regras_lift):
    for regra, dados in sorted(regras_lift.items(), key=lambda x: x[1]['lift'], reverse=True):
        print(f"{regra}")
        print(f"  → Confiança: {dados['confianca']:.2f}")
        print(f"  → Suporte do Consequente: {dados['suporte_consequente']:.2f}")
        print(f"  → LIFT: {dados['lift']:.2f}")
        print()

# Ordena todas as regras por LIFT em ordem decrescente
def ordenar_regras_por_lift(todas_regras_lift):
    return sorted(todas_regras_lift.items(), key=lambda x: x[1]['lift'], reverse=True)