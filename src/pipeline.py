import torch
from collections import Counter
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd
import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import hydra
import logging
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd 
import os
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

def codificar_texto(texto, vocabulario):
    return [vocabulario[palabra] for palabra in texto.split() if palabra in vocabulario]

def pad_secuencias(secuencias, tam_maximo):
    return [seq + [0] * (tam_maximo - len(seq)) if len(seq) < tam_maximo else seq[:tam_maximo] for seq in secuencias]

def obtener_optimizador( nombre , model , lr ):
    if nombre == 'adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif nombre == 'adamw':
        return optim.AdamW( model.parameters(), lr=lr )
    elif nombre == 'adamfactor':
        return optim.Adafactor( model.parameters(), lr=lr  )

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Trabajando con: {device}")

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

    df_balanceado = (
        df_nuevo
        .groupby('clase', group_keys=False)
        .sample(n=5441,random_state=42)
    )

    # Limpieza de datos
    datos_limpios = limpiar_textos( 
        df_balanceado , 
        quitar_mayusculas=cfg.preprocesamiento.quitar_mayusculas, 
        quitar_ascentos=cfg.preprocesamiento.quitar_ascentos , 
        normalizar_simbolos=cfg.preprocesamiento.normalizar_simbolos,
        quitar_numeros=cfg.preprocesamiento.quitar_numeros
    )

    # Procesado y Caracterización
    palabras = [palabra for texto in datos_limpios['texto'] for palabra in texto.split()]
    num_palabras = Counter(palabras)
    vocabulario = {palabra: i+1 for i, (palabra, _) in enumerate(num_palabras.most_common(cfg.procesamiento.max_vocabulary))}
    logger.info(f"Tamaño del vocabulario: {len(vocabulario)}")
    datos_codificados = [codificar_texto(text, vocabulario) for text in datos_limpios['texto']]

    # Enbeddings
    project_root = get_original_cwd()
    data_path = os.path.join(project_root, cfg.embeddings.path)
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

    logger.info(f"Numero de palabras sin reconocer: {numero_palabras_sin_reconocer}")

    weights_matrix = torch.from_numpy(matrix_w2v).float()

    # Preparación de los vectores númericos
    # tam_maximo = max(len(seq) for seq in datos_codificados)
    padded_datos = pad_secuencias(datos_codificados, cfg.hiperparametros.TAM_MAXIMO )

    X = torch.tensor(padded_datos, dtype=torch.long)
    y = torch.tensor(df_balanceado['clase'].values, dtype=torch.long)

    class Dato(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
        
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)

    train_dataset = Dato(X_train, y_train)
    val_dataset = Dato(X_val, y_val)
    test_dataset = Dato(X_test, y_test)

    # BATCH_SIZE = 8

    train_loader = DataLoader(train_dataset, cfg.hiperparametros.BATCH_SIZE , shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= cfg.hiperparametros.BATCH_SIZE )
    test_loader = DataLoader(test_dataset, batch_size=cfg.hiperparametros.BATCH_SIZE)

    class ClasificacionTextoBiLSTM(nn.Module):
        def __init__(self, weights_matrix, hidden_dim , dropout_val ):
            super(ClasificacionTextoBiLSTM, self).__init__()

            self.embedding = nn.Embedding.from_pretrained(weights_matrix)
            self.embedding.weight.requires_grad = False

            self.lstm = nn.LSTM(300, hidden_dim, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, 5)
            # self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(dropout_val)

        def forward(self, x):
            embedded = self.embedding(x)
            _, (hidden, _) = self.lstm(embedded)
            cat_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            cat_hidden = self.dropout(cat_hidden)
            return self.fc(cat_hidden)

    tam_vocabulario = len(vocabulario) + 1

    model = ClasificacionTextoBiLSTM(weights_matrix, cfg.hiperparametros.HIDDEN_DIM , cfg.hiperparametros.DROPOUT ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = obtener_optimizador( nombre= cfg.hiperparametros.OPTIMIZADOR, model=model , lr= cfg.hiperparametros.ls )

    train_losses = []
    val_losses = []

    for epoch in range( cfg.hiperparametros.EPOCAS ):
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device) #GPU
            labels = labels.to(device) #GPU
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device) #GPU
                labels = labels.to(device) #GPU
                outputs = model(inputs)

                val_loss_batch = criterion(outputs, labels)
                total_val_loss += val_loss_batch.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logger.info(f"Época [{epoch+1}/{cfg.hiperparametros.EPOCAS}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    ruta = Path("./Graficas/epoca_vs_perdida.png")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot( range(cfg.hiperparametros.EPOCAS) , train_losses, label='Train Loss')
    plt.plot( range(cfg.hiperparametros.EPOCAS) , val_losses, label='Val Loss')
    plt.title('Curvas de Pérdida durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.savefig(ruta)

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # obtener la clase con mayor probabilidad
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    logger.info("\nResultados en Test:")
    logger.info(f"Accuracy : {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall   : {recall:.4f}")
    logger.info(f"F1 Score : {f1:.4f}")

    logger.info(f"Reporte de clasificación: \n {classification_report(y_true, y_pred)}")

    cm = confusion_matrix(y_true, y_pred)

    ruta = Path("./Graficas/matriz_confusion.png")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.savefig(ruta)

if __name__ == "__main__":
    main()