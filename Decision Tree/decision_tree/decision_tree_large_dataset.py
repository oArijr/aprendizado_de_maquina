"""
#3 - Algoritmo de Árvore de Decisão para uma base de dados maior (Credit Data)

Nesta seção você deverá testar o uso da Árvore de Decisão para a Base de Dados Credit Risk Dataset.
Aqui estaremos analisando os clientes que pagam (classe 0) ou não pagam a dívida (classe 1), a fim do banco conceder empréstimo.
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle

# abrir o arquivo
with open('credit.pkl', 'rb') as f:
  X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

"""a) Ao abrir o arquivo utilize .shape para verificar o tamanho dos dados de treinamento e de teste

OBS: os dados de treinamento devem ter as seguintes dimenções: x=(1500, 3), y=(1500,); os dados de teste devem ter as seguintes dimenções: x=(500, 3), y=(500,)"""

print(f"Treinamento X= {X_credit_treinamento.shape}")
print(f"Treinamento y= {y_credit_treinamento.shape}")
print(f"Teste X= {X_credit_teste.shape}")
print(f"Teste y= {y_credit_teste.shape}")

"""b) Importe o pacote DecisionTreeClassifier do sklearn para treinar o seu algoritmo de árvore de decisão. Para poder refazer os testes e obter o mesmo resultado utilize o parâmetro random_state = 0."""
from sklearn.tree import DecisionTreeClassifier

arvore_credit = DecisionTreeClassifier(criterion='entropy', random_state=0)
arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)

"""c) Faça a previsão com os dados de teste. Visualize os dados e verifique se as previsões estão de acordo com os dados de teste (respostas reais)."""
credit_test_predict = arvore_credit.predict(X_credit_teste)

resultado_df = pd.DataFrame({
    'Real': y_credit_teste,
    'Previsto': credit_test_predict
})

print(resultado_df.head(20))

"""d) Agora faça o cálculo da acurácia para calcular a taxa de acerto entre os valores reais (y teste) e as previsões"""
from sklearn.metrics import accuracy_score, precision_score, recall_score
credit_accuracy = accuracy_score(y_credit_teste, credit_test_predict) * 100

print(f"\nAcurácia do credit: {credit_accuracy}%")

# Precisão
precisao = precision_score(y_credit_teste, credit_test_predict, average='macro')
print(f"\nPrecisão: {precisao * 100:.2f}%")

# Recall
recall = recall_score(y_credit_teste, credit_test_predict, average='macro')
print(f"\nRecall: {recall * 100:.2f}%")

"""e) Faça a análise da Matriz de Confusão.

i. Quantos clientes foram classificados corretamente que pagam a dívida?

ii. Quantos clientes foram classificados incorretamente como não pagantes?

iii. Quantos clientes foram classificados corretamente que não pagam?

iv. Quantos clientes foram classificados incorretamente como pagantes?
"""
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_credit_teste, credit_test_predict)
df_matriz = pd.DataFrame(
    matriz,
    index=["Real 0", "Real 1"],
    columns=["Previsto 0", "Previsto 1"]
)
print("\nMatriz de Confusão:")
print(df_matriz.to_string(), "\n\n")

# i. Foram classificados 430 corretamente como risco alto
# ii. Foram classificados 6 incorretamente como risco baixo
# iii. Foram classificados 61 corretamente como risco baixo.
# iv. Foram classificados 3 incorretamente como risco alto.

"""f) Faça um print do parâmetro classification_report entre os dados de teste e as previsões. Explique qual é a relação entre precision e recall nos dados. Como você interpreta esses dados? """
from sklearn.metrics import classification_report

print(classification_report(y_credit_teste, credit_test_predict))

# Precision indica a proporção de previsões positivas que estavam corretas. No caso da classe 0 (risco alto) a precisão foi de 0.99,
# o que quer dizer que 99% dos clientes previstos como classe 0 (risco alto) realmente são classe 0 (risco alto) e para a classe 1 (risco baixo),
# a precisão foi de 0.91, ou seja, 91% dos clientes previstos como classe 1 (risco baixo) realmente são classe 1 (risco  baixo).

# Recall mostra a proporção de casos positivos reais que foram corretamente identificados pelo modelo.
# Para a classe 0 (risco alto), o recall foi de 0.99, indicando que 99% dos classes 0 foram corretamente detectados.
# Para a classe 1 (risco baixo), o recall foi de 0.95, o que significa que 95% dos classes 1 foram corretamente detectados.

# Aqui vimos que o algoritmo está com uma tendência maior a classificar os clientes como risco baixo.

"""g) Gere uma visualização da sua árvore de decisão utilizando o pacote tree da biblioteca do sklearn.

OBS 1: Os atributos previsores são = ['income', 'age', 'loan']

OBS 2: Adicione cores, nomes para os atributos e para as classes. Você pode utilizar a função fig.savefig para salvar a árvore em uma imagem .png 
"""
from sklearn import tree
atributos = ['income', 'age', 'loan']
classes = ['pagam', 'não pagam']

plt.figure(figsize=(40, 20))
tree.plot_tree(
    arvore_credit,
    feature_names=atributos,
    class_names=classes,
    filled=True,
    proportion=True,
    rounded=True,
    fontsize=12
)
plt.savefig("arvore_credito.png", dpi=300)
plt.show()
