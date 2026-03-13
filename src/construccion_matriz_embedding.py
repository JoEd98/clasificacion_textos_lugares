import gensim
import pickle
import numpy as np
from pathlib import Path
import hydra 
from omegaconf import DictConfig 
import logging
import re
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd 
import os
import pandas as pd

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):

    logger.info("Configuracion: \n" + OmegaConf.to_yaml(cfg))

    # data_path = "./data/caracterizacion/dataset_codificado.pkl"
    project_root = get_original_cwd()
    data_path = os.path.join( project_root, "./data/caracterizacion/" , cfg.procesos.data_path_input )
    with open( data_path , "rb") as f:
        data = pickle.load(f)

    vocabulario = data["vocabulario"]
    datos_codificados = data["datos_codificados"]
    datos_procesados = data["datos_procesados"]

    print(datos_procesados["clase"].unique())

    ## Embeddings
    # data_path = './data/originales/SBW-vectors-300-min5.txt'
    data_path = os.path.join( project_root, "./data/originales/" , cfg.procesos.data_path_embedding )
    modelo_w2v = gensim.models.KeyedVectors.load_word2vec_format(data_path, binary=False) 
    tam_vocabulario = len(vocabulario) + 1
    logger.info( f"Numero de palabras: {tam_vocabulario}" )
    dimension_embedding = 300
    matrix_w2v = np.zeros((tam_vocabulario, dimension_embedding))

    numero_palabras_sin_reconocer = 0
    palabras_no_reconocidas=[]
    for palabra, i in vocabulario.items():
        if palabra in modelo_w2v:
            matrix_w2v[i] = modelo_w2v[palabra]
        else:
            # Si la palabra no existe en el Word2Vec de noticias, vector aleatorio
            matrix_w2v[i] = np.random.normal(scale=0.6, size=(dimension_embedding,))
            numero_palabras_sin_reconocer=numero_palabras_sin_reconocer+1
            palabras_no_reconocidas.append(palabra)
    logger.info( f"Numero de palabras sin reconocer: {numero_palabras_sin_reconocer}" )

    new_data = {
        "vocabulario": vocabulario,
        "datos_codificados": datos_codificados,
        "datos_procesados": datos_procesados,
        "matrix_w2v": matrix_w2v
    }

    nombre_embedding = cfg.procesos.data_path_embedding
    nombre = f"matriz_embeddings_{nombre_embedding[:3]}"
    match = re.search(r"dataset_codificado_(.*?)\.pkl$", cfg.procesos.data_path_input )
    valor = "_" + match.group(1) if match else ""
    nombre += valor
    no_reconocido = nombre + ".csv"
    nombre += ".pkl"

    data_path = os.path.join(project_root, "./data/matriz_embedding/",nombre)
    ruta = Path( data_path )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # ruta_salida = Path("./data/matrices_embeddings/matriz_embeddings.pkl")
    with open(ruta, "wb") as f:
        pickle.dump(new_data, f)

    logger.info( f"PATH_OUTPUT: ./data/matriz_embedding/{nombre}" )

    df = pd.DataFrame(palabras_no_reconocidas, columns=["No_Reconocidas"])
    data_path = os.path.join(project_root, "./data/vocabulario_no_reconocido/",no_reconocido)
    ruta = Path( data_path )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False, encoding='utf-8')

if __name__ == "__main__":
    main()