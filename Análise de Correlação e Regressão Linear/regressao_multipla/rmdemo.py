import numpy as np
import scipy.io as scipy
import sys
import regressao_simples.regressao as reg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# a)
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

# b)
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

print(house_average)
print(smaller_house_price)
print(room_number)


# c)
independent_variables_matrix = [[houses_size[i], houses_room_amount[i]] for i in range(len(houses_size))]
print("\nMatriz de variáveis independentes (tamanho da casa e número de quartos):")
for i in range(len(independent_variables_matrix)):
    print(f"{independent_variables_matrix[i][0]:<18} | {independent_variables_matrix[i][1]:<18}")


# d)
house_datasets = {
    1: houses_size,
    2: houses_room_amount
}
correlacoes = []
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for i, ax in zip(house_datasets, axes):
    # Acessar os dados correspondentes a i (houses_size ou houses_room_amount)
    dados_x = house_datasets[i]

    correlacao = reg.correlacao(dados_x, houses_price)
    correlacoes.append(correlacao)
    regressao_linear = reg.regressao(dados_x, houses_price)
    b0 = reg.calcular_b0(dados_x, houses_price)
    b1 = reg.calcula_b1(dados_x, houses_price)

    ax.scatter(dados_x, houses_price, label=f'Dados {i}')
    ax.plot(dados_x, regressao_linear, label=f"Linha de Regressão {i}", color='red', linewidth=2)

    ax.set_title(f"\nCorrelação: {correlacao:.3f}\nB0: {b0:.3f}\nB1: {b1:.3f}", fontweight='bold')
    ax.set_xlabel("X - Variáveis Independentes", fontsize=12)
    ax.set_ylabel("Y - Variáveis Dependentes", fontsize=12)

#plt.show()


def matrix(list1, list2):
    return [[1, list1[i], list2[i]] for i in range(len(list1))]

def beta(matrix, dependente):
    matrix_transpose = np.matrix_transpose(matrix)
    return (np.linalg.inv(matrix_transpose @ matrix)) @ matrix_transpose @ dependente


def multiple_regression():
    return X * mr


X = matrix(houses_size, houses_room_amount)
mr = beta(X, houses_price)
multiple_regression = multiple_regression()
print(mr)


# e)
x = houses_size
y = houses_room_amount
z = houses_price

plt.ion()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(x, y, z, c='r', marker='o')

b0 = mr[0]
b1 = mr[1] # Coefficient for 'Tamanho'
b2 = mr[2] # Coefficient for 'Quartos'

corr_size_price = correlacoes[0]
corr_rooms_price = correlacoes[1]

# f)
min_x, max_x = min(x), max(x)
min_y, max_y = min(y), max(y)
x_grid, y_grid = np.meshgrid(np.linspace(min_x, max_x), np.linspace(min_y, max_y))

z_grid = b0 + b1 * x_grid + b2 * y_grid

ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, color='blue', label='Plano de Regressão')


ax.set_xlabel('Tamanho da casa')
ax.set_ylabel('Quantidade de quartos')
ax.set_zlabel('Preço')

# g)
equation = f'Preço = {b0:.2f} + {b1:.2f}×(Tamanho) + {b2:.2f}×(Quartos)'
correlation_info = f'Correlação: Tamanho-Preço = {corr_size_price:.3f}, Quartos-Preço = {corr_rooms_price:.3f}'

ax.set_title(equation)

ax.text2D(0.5, 0.01, correlation_info, ha='center', va='bottom', transform=ax.transAxes)

plt.tight_layout()

fig.canvas.draw_idle() # Redraw without freezing
plt.pause(0.1)

plt.show(block=True)


# h)
result = b0 + (b1 * 1650) + (b2 * 3)
print(f"Tamanho de 1650m e 3 quartos : {result:.2f}")
# Ao aumentar a quantidade de quartos o preço da casa diminui, e ao diminuir a quantidade de quartos o preço aumenta.
# A correlação entre o preço e a quantidade de quartos é baixa, resultando em uma regressão ineficaz.
# O coeficiente para o número de quartos (b2) na regressão múltipla é negativa.


# No gráfico 3D, o plano de regressão mostra como o preço (Z) varia em função do tamanho da casa (X) e do número de quartos (Y)