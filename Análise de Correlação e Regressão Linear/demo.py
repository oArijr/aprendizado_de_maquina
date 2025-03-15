import matplotlib.pyplot as plt
import numpy as np
import regressao

x1 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])

x2 = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
y2 = np.array([9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])

x3 = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19])
y3 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50])

plt.figure(figsize=(8, 6))
plt.scatter(x1, y1)

plt.plot(x1, regressao.regressao(x1, y1), label="Linha de Regressão", color='red', linewidth=2)  # Linha de regressão


correlacao = regressao.correlacao(x1, y1)

plt.title("Correlação: " + str(correlacao) + "\nB0: " + str(regressao.calcular_b0(x1, y1)) + "\nB1: " + str(regressao.calcula_b1(x1, y1)), fontweight='bold')  # Título do gráfico
plt.xlabel("X - Variáveis Independentes", fontsize=12)  # Rótulo do eixo X
plt.ylabel("Y - Variáveis Dependentes", fontsize=12)  # Rótulo do eixo Y
plt.show()