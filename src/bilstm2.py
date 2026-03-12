from pathlib import Path
import pickle
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 
from sklearn.model_selection import train_test_split 
from hydra.utils import get_original_cwd 
import hydra 
from omegaconf import DictConfig 
import logging

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

    ruta = Path( cfg.procesos.entrenamiento.path_embedding )
    with open( ruta , "rb") as f:
        matrix_w2v = pickle.load(f)

    ruta = Path( cfg.procesos.entrenamiento.path_dataset_codificado )
    with open( ruta , "rb") as f:
        data = pickle.load(f)

    vocabulario = data["vocabulario"]
    datos_codificados = data["datos_codificados"]

    df = pd.read_csv( cfg.procesos.entrenamiento.path_dataset , encoding='utf-8' )

    weights_matrix = torch.from_numpy(matrix_w2v).float()
    tam_maximo = max(len(seq) for seq in datos_codificados)
    padded_datos = pad_secuencias(datos_codificados, tam_maximo )

    X = torch.tensor(padded_datos, dtype=torch.long)
    y = torch.tensor(df['clase'].values, dtype=torch.long)

    print( X.shape )

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

    train_loader = DataLoader(train_dataset, cfg.procesos.entrenamiento.BATCH_SIZE , shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= cfg.procesos.entrenamiento.BATCH_SIZE )
    test_loader = DataLoader(test_dataset, batch_size=cfg.procesos.entrenamiento.BATCH_SIZE)

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

    model = ClasificacionTextoBiLSTM(weights_matrix, cfg.procesos.entrenamiento.HIDDEN_DIM , cfg.procesos.entrenamiento.DROPOUT ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = obtener_optimizador( nombre= cfg.procesos.entrenamiento.OPTIMIZADOR, model=model , lr= cfg.procesos.entrenamiento.ls )

    train_losses = []
    val_losses = []

    for epoch in range( cfg.procesos.entrenamiento.EPOCAS ):
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

        # logger.info(f"Época [{epoch+1}/{cfg.procesos.entrenamiento.EPOCAS}] -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


if __name__ == "__main__":
    main()