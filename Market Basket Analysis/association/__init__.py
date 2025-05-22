import pandas as pd
from apyori import apriori
import utils

limiar_suporte = 0.3
fator_confianca = 0.8

base_mercado = pd.read_csv('mercado.csv', header = None)
print(base_mercado)

todos_itens = base_mercado.values.flatten()
todos_itens = [item for item in todos_itens if pd.notna(item)] # Remove NaN

itens_unicos = set(todos_itens)
qtd = len(base_mercado)

suportes = []
for item in itens_unicos:
    contagem = todos_itens.count(item)
    suposto_suporte = contagem / qtd
    if suposto_suporte >= limiar_suporte:
        suportes.append((item, suposto_suporte))
print(suportes)

itens_validos = set(item for item, suporte in suportes)
base_filtrada = base_mercado.where(base_mercado.isin(itens_validos))

duplas_resultado = utils.combinacoes_duplas(base_filtrada)

trios_resultado = utils.combinacoes_trios(base_filtrada)

dupla_confianca = utils.regras_calculo_duplas(duplas_resultado, suportes, fator_confianca)

utils.exibir(duplas_resultado)

utils.exibir(trios_resultado)

utils.exibir(dupla_confianca)