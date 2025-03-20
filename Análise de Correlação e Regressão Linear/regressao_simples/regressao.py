import numpy as np
import math

x1 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])

x2 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
y2 = np.array([9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])

x3 = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19])
y3 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50])


def correlacao(vetor_x, vetor_y):
    mean_x = np.mean(vetor_x)
    mean_y = np.mean(vetor_y)

    total_sum = 0
    for x, y in zip(vetor_x, vetor_y):
        soma_x = x - mean_x
        soma_y = y - mean_y
        total_sum += soma_x * soma_y

    soma_x_squared = 0
    soma_y_squared = 0
    for x, y in zip(vetor_x, vetor_y):
        soma_x_squared += pow(x - mean_x, 2)
        soma_y_squared += pow(y - mean_y, 2)
    total_squared_sum = soma_x_squared * soma_y_squared

    down_part = math.sqrt(total_squared_sum)

    return total_sum / down_part

def regressao(vetor_x, vetor_y):
    mean_x = np.mean(vetor_x)
    mean_y = np.mean(vetor_y)

    total_sum = 0
    for x, y in zip(vetor_x, vetor_y):
        soma_x = x - mean_x
        soma_y = y - mean_y
        total_sum += soma_x * soma_y

    soma_x_squared = 0
    for x, y in zip(vetor_x, vetor_y):
        soma_x_squared += pow(x - mean_x, 2)

    b1 = total_sum / soma_x_squared

    b0 = mean_y - (b1 * mean_x)

    if (__name__ == "__main__"):
        print("BO e B1:")
        print(b0)
        print(b1)

    response = []
    for x in vetor_x:
        response.append(b0 + (b1 * x))

    return response

def calcula_b1(x, y):
    media_x = np.mean(x)
    media_y = np.mean(y)

    desvio_x = x - media_x
    desvio_y = y - media_y

    produto_x_y = desvio_x * desvio_y

    somatoria_x_y = sum(produto_x_y)

    denominador = sum((desvio_x ** 2))

    return somatoria_x_y / denominador


def calcular_b0(x, y):
    media_x = np.mean(x)
    media_y = np.mean(y)
    b1 = calcula_b1(x, y)
    return media_y - (b1 * media_x)

if (__name__ == "__main__"):
    print(correlacao(x1, y1))
    print(regressao(x1, y1))
    print()

    print(correlacao(x2, y2))
    print(regressao(x2, y2))
    print()

    print(correlacao(x3, y3))
    print(regressao(x3, y3))
