import pandas as pd
from apyori import apriori
import utils

# Parâmetros
limiar_suporte = 0.3
fator_confianca = 0.8

# Carrega a base de dados
base_mercado = pd.read_csv('mercado.csv', header=None)
print("Base de Dados:")
print(base_mercado)

# Obtém todos os itens e remove NaN
todos_itens = base_mercado.values.flatten()
todos_itens = [item for item in todos_itens if pd.notna(item)]

# Obtém itens únicos e calcula o suporte
itens_unicos = set(todos_itens)
qtd = len(base_mercado)  # Total de transações = 10

# Calcula o suporte para cada item
suportes = []
for item in itens_unicos:
    contagem = todos_itens.count(item)
    suposto_suporte = contagem / qtd
    if suposto_suporte >= limiar_suporte:
        suportes.append((item, suposto_suporte))
print('\nSuportes (suporte >= 0.3):', suportes)

# Converte suportes em um dicionário
suportes_dict = {item: suporte for item, suporte in suportes}

# Filtra a base para conter apenas itens com suporte suficiente
itens_validos = set(item for item, suporte in suportes)
base_filtrada = base_mercado.where(base_mercado.isin(itens_validos))

# Gera combinações de duplas
duplas_resultado = utils.combinacoes_duplas(base_filtrada)
print('\nDuplas:')
utils.exibir(duplas_resultado)

# Gera combinações de trios
trios_resultado = utils.combinacoes_trios(base_filtrada)
print('\nTrios:')
utils.exibir(trios_resultado)

# Calcula regras de associação para duplas com confiança e lift
regras_duplas = utils.regras_calculo_duplas(duplas_resultado, suportes_dict, fator_confianca)
print(f'\nRegras de Duplas com Confiança >= {fator_confianca}:')
utils.exibir(regras_duplas)

trios_confianca, itemsets_validos = utils.regras_calculo_trios(trios_resultado, duplas_resultado, suportes_dict, qtd, fator_confianca)

# Exibe as regras de trios
print(f'\nRegras de Trios com Confiança >= {fator_confianca}:')
utils.exibir_regras_trios(trios_confianca)
