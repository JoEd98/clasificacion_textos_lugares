import pandas as pd
from collections import Counter

def codificar_texto(texto, vocabulario):
    return [vocabulario[palabra] for palabra in texto.split() if palabra in vocabulario]

def main():
    data_path = './data/procesados/procesados_minus.csv'
    df = pd.read_csv( data_path , encoding='utf-8')

    # Procesado y Caracterización
    palabras = [palabra for texto in df['texto'] for palabra in texto.split()]
    num_palabras = Counter(palabras)
    vocabulario = {palabra: i+1 for i, (palabra, _) in enumerate(num_palabras.most_common())}
    print(vocabulario)
    # logger.info(f"Tamaño del vocabulario: {len(vocabulario)}")
    datos_codificados = [codificar_texto(text, vocabulario) for text in df['texto']]

    print(datos_codificados[0])

if __name__ == "__main__":
    main()