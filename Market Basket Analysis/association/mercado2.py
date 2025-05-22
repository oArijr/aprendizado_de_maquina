import pandas as pd
from apyori import apriori
import utils

# Carrega a base de dados
base_mercado = pd.read_csv('mercado2.csv', header=None)
print("Primeiros 10 registros:")
print(base_mercado.head(10))



# A)
print(f"\n{'=' * 80}")
print("QUESTÃO A: Produtos vendidos pelo menos 4 vezes ao dia (28 vezes por semana)")

# Obtém todos os itens e remove NaN
todos_itens = base_mercado.values.flatten()
todos_itens = [item for item in todos_itens if pd.notna(item)]

# Define o limiar de suporte com base no critério de 28 vendas por semana
vendas_semanais_min = 28
qtd_transacoes = len(base_mercado)
limiar_suporte = vendas_semanais_min / qtd_transacoes

print(f"Critério: produtos vendidos pelo menos {vendas_semanais_min} vezes por semana")
print(f"Total de transações: {qtd_transacoes}")
print(f"Limiar de suporte: {limiar_suporte:.4f}")

# Obtém itens únicos e calcula suporte
itens_unicos = set(todos_itens)
suportes = []
for item in itens_unicos:
    contagem = todos_itens.count(item)
    suporte_item = contagem / qtd_transacoes
    if suporte_item >= limiar_suporte:
        suportes.append((item, suporte_item, contagem))

# Exibe os suportes válidos
print(f"\nItens com suporte >= {limiar_suporte:.4f}:")
suportes.sort(key=lambda x: x[1], reverse=True)
for item, suporte, contagem in suportes:
    print(f"  {item}: suporte = {suporte:.4f} ({contagem} ocorrências)")

print(f"\nRESULTADO QUESTÃO A: {len(suportes)} produtos atendem ao critério")



# B)
print(f"\n{'=' * 80}")
print("QUESTÃO B: Aplicando Confiança = 0.2 e LIFT = 3")

fator_confianca = 0.2
limiar_lift = 3.0

print("Parâmetros configurados:")
print(f"  - Suporte mínimo: {limiar_suporte:.4f}")
print(f"  - Confiança mínima: {fator_confianca}")
print(f"  - LIFT mínimo: {limiar_lift}")

# Converte suportes em dicionário
suportes_dict = {item: suporte for item, suporte, _ in suportes}

# Filtra base para manter apenas itens válidos
itens_validos = set(item for item, _, _ in suportes)
base_filtrada = base_mercado.where(base_mercado.isin(itens_validos))

print(f"\nProcessando combinações com {len(itens_validos)} itens válidos...")

# Gera combinações de duplas
duplas_resultado = utils.combinacoes_duplas(base_filtrada)
print(f"Duplas encontradas: {len(duplas_resultado)}")

# Gera combinações de trios
trios_resultado = utils.combinacoes_trios(base_filtrada)
print(f"Trios encontrados: {len(trios_resultado)}")

# Calcula regras de duplas com confiança
regras_duplas = utils.regras_calculo_duplas(duplas_resultado, suportes_dict, fator_confianca)
print(f"Regras de duplas com confiança >= {fator_confianca}: {len(regras_duplas)}")

# Calcula regras de trios com confiança
trios_confianca, itemsets_validos = utils.regras_calculo_trios(
    trios_resultado, duplas_resultado, suportes_dict, qtd_transacoes, fator_confianca)
print(f"Regras de trios com confiança >= {fator_confianca}: {len(trios_confianca)}")

# Calcula LIFT para duplas
lift_duplas = utils.calcular_lift_duplas(regras_duplas, suportes_dict)
regras_duplas_lift_validas = {k: v for k, v in lift_duplas.items() if v['lift'] >= limiar_lift}

# Calcula LIFT para trios
lift_trios = utils.calcular_lift_trios(trios_confianca, suportes_dict, duplas_resultado)
regras_trios_lift_validas = {k: v for k, v in lift_trios.items() if v['lift'] >= limiar_lift}

print(f"\nRegras de duplas com LIFT >= {limiar_lift}: {len(regras_duplas_lift_validas)}")
print(f"Regras de trios com LIFT >= {limiar_lift}: {len(regras_trios_lift_validas)}")

total_regras_validas = len(regras_duplas_lift_validas) + len(regras_trios_lift_validas)
print(f"\nRESULTADO QUESTÃO B: {total_regras_validas} regras foram retornadas")



# C)
print(f"\n{'=' * 80}")
print("QUESTÃO C: Visualização dos dados usando algoritmo Apriori")

# Prepara transações para o algoritmo Apriori (sem remover NaNs)
transacoes = []
for _, linha in base_mercado.iterrows():
    transacao = [item for item in linha if pd.notna(item)]
    if transacao:
        transacoes.append(transacao)

print(f"Preparando dados para Apriori: {len(transacoes)} transações")

# Executa o algoritmo Apriori
print("Executando algoritmo Apriori...")
resultados_apriori = list(apriori(transacoes,
                                  min_support=limiar_suporte,
                                  min_confidence=fator_confianca,
                                  min_lift=limiar_lift,
                                  max_length=3))

# Exibe os resultados do Apriori
print(f"Quantidade de regras geradas: {len(resultados_apriori)}")

# Inicializa listas vazias para armazenar os antecedentes (A), consequentes (B), suporte, confiança e lift das regras
A = []  # Lista para armazenar os antecedentes (itens à esquerda da regra)
B = []  # Lista para armazenar os consequentes (itens à direita da regra)
suporte = []  # Lista para armazenar os valores de suporte das regras
confianca = []  # Lista para armazenar os valores de confiança das regras
lift = []  # Lista para armazenar os valores de lift das regras

# Itera sobre os resultados gerados pelo algoritmo Apriori
for resultado in resultados_apriori:  # Cada 'resultado' contém um itemset e suas regras de associação
    s = resultado[1]  # Extrai o valor de suporte do itemset (frequência relativa do itemset)
    result_rules = resultado[2]  # Extrai as regras de associação associadas ao itemset
    for result_rule in result_rules:  # Itera sobre cada regra dentro do itemset
        a = list(result_rule[0])  # Converte o conjunto de antecedentes (itens à esquerda) em uma lista
        b = list(result_rule[1])  # Converte o conjunto de consequentes (itens à direita) em uma lista
        c = result_rule[2]  # Extrai o valor de confiança da regra
        l = result_rule[3]  # Extrai o valor de lift da regra
        A.append(a)  # Adiciona os antecedentes à lista A
        B.append(b)  # Adiciona os consequentes à lista B
        suporte.append(s)  # Adiciona o suporte à lista suporte
        confianca.append(c)  # Adiciona a confiança à lista confianca
        lift.append(l)  # Adiciona o lift à lista lift

# Cria um DataFrame com as listas A, B, suporte, confiança e lift para organizar os resultados
rules_df = pd.DataFrame({'A': A, 'B': B, 'suporte': suporte, 'confianca': confianca, 'lift': lift})

# Ordena o DataFrame pelo valor de lift em ordem decrescente para destacar as regras mais relevantes
rules_df = rules_df.sort_values(by='lift', ascending=False)

# Exibe o DataFrame com as regras ordenadas
print("\nRegras de associação ordenadas por lift:")
print(rules_df)



# D)
print(f"\n{'=' * 80}")
print("QUESTÃO D: Ordenando pelo valor de confiança para encontrar a regra mais confiável")

# Ordena o DataFrame pelo valor de confiança em ordem decrescente
rules_df = rules_df.sort_values(by='confianca', ascending=False)

# Exibe o DataFrame ordenado
print("\nRegras ordenadas por confiança (maior para menor):")
print(rules_df)

# Identifica a regra com maior confiança
maior_confianca = rules_df['confianca'].iloc[0]
regra_maior_confianca = rules_df.iloc[0]

print(f"\nRESULTADO QUESTÃO D:")
print(f"A regra com maior confiabilidade é: {regra_maior_confianca['A']} -> {regra_maior_confianca['B']}")
print(f"Valor da maior confiança: {maior_confianca:.4f}")