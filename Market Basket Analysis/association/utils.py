import pandas as pd

def combinacoes_duplas(base_mercado):
    for index, row in base_mercado.iterrows():
        for item in row:
            if pd.notna(item): # ignora NaNs
                print(f"Item da linha {index}: {item}")
                print(base_mercado[index][item])