import pandas as pd
import pickle
from collections import Counter
from pathlib import Path

def codificar_texto(texto, vocabulario):
    tokens = texto.split()
    return len(tokens) , [vocabulario[palabra] for palabra in tokens if palabra in vocabulario]

def main():
    data_path = './data/procesados/procesados_30_balan_minus_acentos.csv'
    df = pd.read_csv( data_path , encoding='utf-8')

    # Procesado y Caracterización
    palabras = [palabra for texto in df['texto'] for palabra in texto.split()]
    num_palabras = Counter(palabras)
    vocabulario = {palabra: i+1 for i, (palabra, _) in enumerate(num_palabras.most_common())}
    print( len(vocabulario) )
    # logger.info(f"Tamaño del vocabulario: {len(vocabulario)}")
    # datos_codificados = [codificar_texto(text, vocabulario) for text in df['texto']]

    datos_codificados = []
    mayor_tam_parrafo = 0
    for text in df['texto']:
        tam_parrafo , vector = codificar_texto(text, vocabulario)
        datos_codificados.append(vector)
        if mayor_tam_parrafo < tam_parrafo:
            mayor_tam_parrafo = tam_parrafo 

    print(mayor_tam_parrafo)

    data = {
        "vocabulario": vocabulario,
        "datos_codificados": datos_codificados
    }

    data_path = "./data/caracterizacion/dataset_codificado.pkl"
    ruta = Path( data_path )
    ruta.parent.mkdir(parents=True, exist_ok=True)

    with open("./data/caracterizacion/dataset_codificado.pkl", "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    main()