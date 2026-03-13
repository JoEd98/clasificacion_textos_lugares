import pandas as pd
import pickle
from collections import Counter
from pathlib import Path
import hydra 
from omegaconf import DictConfig 
import logging
import re
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd 
import os

def codificar_texto(texto, vocabulario):
    tokens = texto.split()
    return len(tokens) , [vocabulario[palabra] for palabra in tokens if palabra in vocabulario]

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):

    logger.info("Configuracion: \n" + OmegaConf.to_yaml(cfg))

    project_root = get_original_cwd()
    data_path = os.path.join( project_root, "./data/procesados/" , cfg.procesos.data_path )
    df = pd.read_csv( data_path , encoding='utf-8')

    # Procesado y Caracterización
    palabras = [palabra for texto in df['texto'] for palabra in texto.split()]
    num_palabras = Counter(palabras)
    vocabulario = {palabra: i+1 for i, (palabra, _) in enumerate(num_palabras.most_common())}
    logger.info(f"Tamaño del vocabulario: {len(vocabulario)}")

    datos_codificados = []
    mayor_tam_parrafo = 0
    for text in df['texto']:
        tam_parrafo , vector = codificar_texto(text, vocabulario)
        datos_codificados.append(vector)
        if mayor_tam_parrafo < tam_parrafo:
            mayor_tam_parrafo = tam_parrafo 
    logger.info(f"Tamaño mas grande de los parrafos: {mayor_tam_parrafo}")

    data = {
        "vocabulario": vocabulario,
        "datos_codificados": datos_codificados,
        "datos_procesados": df
    }

    nombre = "dataset_codificado"
    match = re.search(r"procesados_(.*?)\.csv$", cfg.procesos.data_path )
    valor = "_" + match.group(1) if match else ""
    nombre += valor
    nombre += ".pkl"

    data_path = os.path.join(project_root, "./data/caracterizacion/",nombre)
    ruta = Path( data_path )
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(data, f)

    logger.info( f"PATH_OUTPUT: ./data/caracterizacion/{nombre}" )

if __name__ == "__main__":
    main()