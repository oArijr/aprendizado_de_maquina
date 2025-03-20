import numpy as np
import scipy.io as scipy
import sys
import regressao_simples.regressao as reg
import matplotlib.pyplot as plt

mat = scipy.loadmat('data.mat')
data = mat['data']


total_house_price = 0
total_house_number = len(data)

smaller_house_price = 0
max_int = sys.maxsize


min_int = -1
room_number = 0

houses_size = []
houses_room_amount = []
houses_price = []
for x in range(total_house_number):
    house_size = data[x][0]
    room_amount = data[x][1]
    house_price = data[x][2]

    houses_size.append(house_size)
    houses_room_amount.append(room_amount)
    houses_price.append(house_price)

    total_house_price += house_price
    if max_int > house_size:
        max_int = house_size
        smaller_house_price = house_price
    if min_int < house_price:
        min_int = house_price
        room_number = room_amount

house_average = total_house_price / total_house_number

#b)
print(house_average)
print(smaller_house_price)
print(room_number)


independent_variables_matrix = [[houses_size[i], houses_room_amount[i]] for i in range(len(houses_size))]

# c
print("\nMatriz de variáveis independentes (tamanho da casa e número de quartos):")
for i in range(len(independent_variables_matrix)):
    print(f"{independent_variables_matrix[i][0]:<20} | {independent_variables_matrix[i][1]:<18}")


# d
datasets = {
    1: houses_size,
    2: houses_room_amount
}
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for i, ax in zip(datasets, axes):
    # Acessar os dados correspondentes a i (houses_size ou houses_room_amount)
    dados_x = datasets[i]

    # Calcular correlação e regressão com a função do regressor (presumindo que reg é um objeto de regressão)
    correlacao = reg.correlacao(dados_x, houses_price)
    regressao_linear = reg.regressao(dados_x, houses_price)
    b0 = reg.calcular_b0(dados_x, houses_price)
    b1 = reg.calcula_b1(dados_x, houses_price)

    # Plotar gráfico
    ax.scatter(dados_x, houses_price, label=f'Dados {i}')
    ax.plot(dados_x, regressao_linear, label=f"Linha de Regressão {i}", color='red', linewidth=2)

    ax.set_title(f"\nCorrelação: {correlacao:.3f}\nB0: {b0:.3f}\nB1: {b1:.3f}", fontweight='bold')
    ax.set_xlabel("X - Variáveis Independentes", fontsize=12)
    ax.set_ylabel("Y - Variáveis Dependentes", fontsize=12)

#plt.show()


def matrix(list1, list2):
    return [[1, list1[i], list2[i]] for i in range(len(list1))]

def multiple_regression(matrix, dependente):
    matrix_transpose = np.matrix_transpose(matrix)
    return (np.linalg.inv(matrix_transpose @ matrix)) @ matrix_transpose @ dependente


b = matrix(houses_size, houses_room_amount)
print(b)
print(multiple_regression(b, houses_price))



