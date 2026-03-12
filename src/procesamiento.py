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

def limpiar_textos( 
    datos , 
    quitar_mayusculas=False , 
    quitar_ascentos=False ,
    normalizar_simbolos=False,
    quitar_numeros=False
):
    
    datos_limpios = datos.copy()

    datos_limpios['texto'] = ( datos_limpios['texto'].str.replace(r'[\r\n]+', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip())

    if normalizar_simbolos:
        # Quitar todos los símbolos excepto ! ¡ ? ¿
        datos_limpios['texto'] = datos_limpios['texto'].str.replace(
            r"[^\w\s!¡?¿]", "", regex=True
        )

        # Reducir múltiples !
        datos_limpios['texto'] = datos_limpios['texto'].str.replace(
            r"!+", "!", regex=True
        )

        # Reducir múltiples ?
        datos_limpios['texto'] = datos_limpios['texto'].str.replace(
            r"\?+", "?", regex=True
        )

        # Separar ¿ y ¡
        datos_limpios['texto'] = datos_limpios['texto'].str.replace(
            r"[¡¿!?]+", lambda x: x.group(0)[0] + " ", regex=True
        )

        # Agregar espacio antes de ! o ?
        datos_limpios['texto'] = datos_limpios['texto'].str.replace(
            r"\s*([¡¿!?])", r" \1", regex=True
        )

    if quitar_mayusculas:
        datos_limpios['texto'] = datos_limpios['texto'].str.lower()

    if quitar_ascentos:
        datos_limpios['texto'] = ( datos_limpios['texto'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))

    if quitar_numeros:
        datos_limpios['texto'] = datos_limpios['texto'].str.replace(r"\d+", "", regex=True)

    return datos_limpios

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):

    logger.info("Configuracion: \n" + OmegaConf.to_yaml(cfg))

    params = cfg.procesos.procesamiento

    #Leer archivo

    project_root = get_original_cwd()
    data_path = os.path.join(project_root, params.path_data_input)
    df = pd.read_csv( data_path , encoding='utf-8')

    df = limpiar_textos( 
        df , 
        quitar_mayusculas=params.quitar_mayusculas, 
        quitar_ascentos=params.quitar_ascentos , 
        normalizar_simbolos=params.normalizar_simbolos,
        quitar_numeros=params.quitar_numeros
    )

    config_limpieza = {
        "minus": params.quitar_mayusculas,
        "acentos": params.quitar_ascentos,
        "simbolos": params.normalizar_simbolos,
        "num": params.quitar_numeros,
        "lemma": params.lemmatizar,
        "stem": params.stemming
    }

    nombre = "procesados"

    match = re.search(r"preprocesados_(.*?)\.csv$", params.path_data_input)
    valor = "_" + match.group(1) if match else ""

    nombre += valor

    for key, value in config_limpieza.items():
        if value:
            nombre += f"_{key}"

    nombre += ".csv"

    data_path = os.path.join(project_root, "./data/procesados" , nombre)
    ruta = Path( data_path)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(ruta, index=False, encoding='utf-8')

if __name__ == "__main__":
    main()