import os
import sys
import hydra 
import logging 
import pandas as pd
from pathlib import Path
from ftfy import fix_text
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import get_original_cwd 

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):

    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    logger.info("Configuración: \n" + OmegaConf.to_yaml(cfg))

    project_root = get_original_cwd()
    data_path = os.path.join(project_root, cfg.procesos.path_input)
    df = pd.read_csv( data_path , encoding='utf-8')

    if cfg.procesos.numero_renglones != 'all':
        df = df.head( cfg.procesos.numero_renglones ).copy()

    logger.info(f"Datos antes de la reparación: \n {df.head(5)}")

    for col in df.select_dtypes(include=['object','string']).columns:
        df[col] = df[col].apply(lambda x: fix_text(x) if isinstance(x,str) else x)

    logger.info(f"Datos después de la reparación: \n {df.head(5)}")

    nombre_archivo = "reparados_all.csv" if cfg.procesos.numero_renglones == 'all' else f"reparados_{cfg.procesos.numero_renglones}.csv"

    project_root = get_original_cwd()
    data_path = os.path.join(project_root, "./data/reparados",nombre_archivo)
    ruta = Path( data_path)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(ruta, index=False, encoding='utf-8')

if __name__ == "__main__":
    main()