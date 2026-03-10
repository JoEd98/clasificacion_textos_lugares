import torch.optim as optim
import logging
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import hydra
import pandas as pd 
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

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

def obtener_vectorizador( metodo="tfidf", ngrama=(1,1)):
    if metodo == "binary":
        return CountVectorizer(binary=True, ngram_range=ngrama)
    elif metodo == "tf":
        return CountVectorizer(binary=False, ngram_range=ngrama)
    elif metodo == "tfidf":
        return TfidfVectorizer(ngram_range=ngrama)
    else:
        raise ValueError("Método no válido")

def obtener_modelo(nombre_modelo = 'knn' , parametros = { 'alpha' : 0.01 , 'weights':'uniform' , 'metric':'minkowski' } ):
    if nombre_modelo == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights=parametros['weights'], metric=parametros['metric'])
    if nombre_modelo == "svm":
        return SVC(kernel='linear')
    if nombre_modelo == "bayes":
        return MultinomialNB( alpha=parametros['alpha'] )
    if nombre_modelo == "perseptron":
        return Perceptron(tol=1e-3)
    if nombre_modelo == "arboles":
        return DecisionTreeClassifier()

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info("Configuracion: \n" + OmegaConf.to_yaml(cfg))

    #Leer archivo
    project_root = get_original_cwd()
    data_path = os.path.join(project_root, cfg.data.path)
    df = pd.read_csv( data_path , encoding='utf-8')

    df_nuevo = pd.DataFrame({
        'texto': df['Title'].fillna('') + ' ' + df['Review'].fillna(''),
        'clase': df['Polarity']
    })

    df_nuevo['clase'] = df_nuevo['clase'] - 1

    if cfg.preprocesamiento.balancear_datos:
        df_nuevo = (
            df_nuevo
            .groupby('clase', group_keys=False)
            .sample(n= cfg.preprocesamiento.tam_datos_balanceados,random_state=42)
        )

    logger.info(f"Numero de Textos: {df_nuevo.shape}")

    # Separar datos de entrenamiento, validacion y prueba
    # texto_entrenamiento_validacion, texto_prueba, Y_entrenamiento_validacion, Y_prueba = train_test_split(
    #     df_nuevo['texto'].values , df_nuevo['clase'].values, test_size=0.2, stratify=df_nuevo['clase'].values, random_state=42
    # )

    # texto_entrenamiento, texto_validacion, y_entrenamiento, y_validacion = train_test_split(
    #     texto_entrenamiento_validacion, Y_entrenamiento_validacion, test_size=0.125, stratify=Y_entrenamiento_validacion , random_state=42
    # )

    texto_entrenamiento, texto_prueba, Y_entrenamiento, Y_prueba = train_test_split(
        df_nuevo['texto'].values , df_nuevo['clase'].values, test_size=0.2, stratify=df_nuevo['clase'].values, random_state=42
    )

    vectorizador = obtener_vectorizador( metodo=cfg.experimentos.hiperparametros.metodo , ngrama=( cfg.experimentos.hiperparametros.ngrama_ini , cfg.experimentos.hiperparametros.ngrama_fin ) )
    v_entrenamiento = vectorizador.fit_transform( texto_entrenamiento )
    v_validacion = vectorizador.transform( texto_prueba )

    # Selección de características
    selector = SelectKBest(chi2, k=cfg.experimentos.hiperparametros.k)
    X_entrenamiento_seleccionadas = selector.fit_transform(v_entrenamiento, Y_entrenamiento)
    X_validacion_seleccionadas = selector.transform(v_validacion)

    # Seleccion de modelo y entrenamiento
    modelo = obtener_modelo(nombre_modelo=cfg.experimentos.hiperparametros.nombre_modelo)
    modelo.fit( X_entrenamiento_seleccionadas , Y_entrenamiento )
    y_prediccion = modelo.predict(X_validacion_seleccionadas)

    from sklearn.metrics import f1_score, precision_score , accuracy_score
    score = f1_score(Y_prueba, y_prediccion,average='weighted')
    presicion = precision_score(Y_prueba, y_prediccion,average='weighted')
    accuracy = accuracy_score(Y_prueba, y_prediccion)

    logger.info(f"Score: {score}")
    logger.info(f"Presicion: {presicion}")
    logger.info(f"Accuracy: {accuracy}")

    from sklearn.metrics import classification_report, confusion_matrix

    logger.info( f"Reporte: \n {classification_report(Y_prueba, y_prediccion,zero_division=1)}")

    ruta = Path("./Graficas/matriz_confusion.png")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(Y_prueba, y_prediccion)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.savefig(ruta)

if __name__ == "__main__":
    main()