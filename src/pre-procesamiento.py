import os
import re
import hydra 
import logging 
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import get_original_cwd 

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):

    logger.info("Configuracion: \n" + OmegaConf.to_yaml(cfg))
    params = cfg.procesos.preprocesamiento

    #Leer archivo

    project_root = get_original_cwd()
    data_path = os.path.join(project_root, "./data/reparados/" , params.path_data_input)
    df = pd.read_csv( data_path , encoding='utf-8')

    # Extraer columnas importantes

    df = pd.DataFrame({
        'texto': df['Title'].fillna('') + ' ' + df['Review'].fillna(''),
        'clase': df['Polarity']
    })

    print(df["clase"].unique())

    df['clase'] = df['clase'] - 1

    print(df["clase"].unique())

    # Agregar nuevos datos
    if params.agregar_datos:
        # project_root = get_original_cwd()
        data_path = os.path.join(project_root, "./data/originales/" , params.path_oversampling_input)
        print(data_path)
        df_oversamplig = pd.read_csv( data_path , encoding='utf-8')
        df = pd.concat([ df , df_oversamplig ], ignore_index=True)

    # Balancer datos por clase

    if params.balancear_clases:
        min_clase = df['clase'].value_counts().min()
        df = df.groupby('clase', group_keys=False).sample(n=min_clase,random_state=42)

    logger.info( f"Informacion de cada clase: \n{df['clase'].value_counts()}" )

    conf_prepro = {
        'balan': params.balancear_clases,
        'over': params.agregar_datos
    }

    nombre = "preprocesados"

    match = re.search(r"_(.*?)\.csv$", params.path_data_input)
    valor = "_" + match.group(1) if match else ""

    nombre += valor

    for key, value in conf_prepro.items():
        if value:
            nombre += f"_{key}"

    nombre += ".csv"

    project_root = get_original_cwd()
    data_path = os.path.join(project_root, "./data/pre-procesados/",nombre)
    ruta = Path( data_path)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(ruta, index=False, encoding='utf-8')

    logger.info( f"data output: ./data/pre-procesados/{nombre}" )

if __name__ == "__main__":
    main()