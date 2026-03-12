import gensim
import pickle
import numpy as np

def main():

    
    with open(".\data\caracterizacion\dataset_codificado.pkl", "rb") as f:
        data = pickle.load(f)

    vocabulario = data["vocabulario"]
    datos_codificados = data["datos_codificados"]

    # Enbeddings
    data_path = '.\data\originales\SBW-vectors-300-min5.csv'
    modelo_w2v = gensim.models.KeyedVectors.load_word2vec_format(data_path, binary=False) 
    tam_vocabulario = len(vocabulario) + 1
    dimension_embedding = 300
    matrix_w2v = np.zeros((tam_vocabulario, dimension_embedding))

    numero_palabras_sin_reconocer = 0
    for palabra, i in vocabulario.items():
        if palabra in modelo_w2v:
            matrix_w2v[i] = modelo_w2v[palabra]
        else:
            # Si la palabra no existe en el Word2Vec de noticias, vector aleatorio
            matrix_w2v[i] = np.random.normal(scale=0.6, size=(dimension_embedding,))
            numero_palabras_sin_reconocer=numero_palabras_sin_reconocer+1