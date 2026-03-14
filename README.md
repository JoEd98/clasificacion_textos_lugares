# Clasificación de textos

## Procesos

### Reparar archivos

Reparar todos los registros del archivo:

```bash
python .\src\reparar_archivo.py procesos=reparar_archivos 
```

Reparar solo N registros del archivo:

```bash
python .\src\reparar_archivo.py procesos=reparar_archivos procesos.numero_renglones=3
```

### Pre-procesamiento de datos

Obtener columna de texto y columna clase sin agregar datos y sin balancear clases

Obtener columna de texto y columna clase sin agregar datos y con balancear clases

```bash
python .\src\pre-procesamiento.py procesos=preprocesamiento procesos/preprocesamiento=undersampling procesos.preprocesamiento.path_data_input=reparados_all.csv
```

```bash
python .\src\pre-procesamiento.py procesos=preprocesamiento procesos/preprocesamiento=simple
```

Obtener columna de texto y columna clase agregando datos y balancear clases

```bash
python .\src\pre-procesamiento.py procesos=preprocesamiento procesos/preprocesamiento=oversampling
```

### Procesamiento de datos

```bash
python .\src\procesamiento.py procesos=procesamiento
```

```bash
python src/procesamiento.py procesos=procesamiento procesos/procesamiento=simple procesos.procesamiento.quitar_mayusculas=true procesos.procesamiento.quitar_ascentos=true procesos.procesamiento.normalizar_simbolos=true procesos.procesamiento.quitar_numeros=true procesos.procesamiento.lemmatizar=false procesos.procesamiento.quitar_palabras_vacias=false procesos.procesamiento.path_data_input=preprocesados_all_balan.csv
```

```bash
python src/procesamiento.py --multirun procesos=procesamiento procesos/procesamiento=simple procesos.procesamiento.quitar_mayusculas=true,false procesos.procesamiento.quitar_ascentos=true,false procesos.procesamiento.normalizar_simbolos=true,false procesos.procesamiento.quitar_numeros=true,false procesos.procesamiento.lemmatizar=true,false procesos.procesamiento.quitar_palabras_vacias=true,false procesos.procesamiento.path_data_input=preprocesados_all_balan.csv
```

### Caracterización


```bash
python .\src\caracterizacion.py procesos=caracterizacion procesos.data_path=procesados_all_balan_minus_acentos_simbolos_num.csv
```

```bash
python .\src\caracterizacion.py procesos=caracterizacion procesos.data_path=procesados_all_balan_acentos_simbolos_num_lemma.csv
```

```bash
python .\src\caracterizacion.py procesos=caracterizacion procesos.data_path=procesados_all_balan_minus_simbolos_num_lemma_stem.csv
```


```bash
python .\src\caracterizacion.py --multirun procesos=caracterizacion procesos.data_path=procesados_all_balan_minus_acentos_simbolos.csv,procesados_all_balan_minus_acentos_simbolos_lemma.csv,procesados_all_balan_minus_acentos_simbolos_lemma_palVac.csv,procesados_all_balan_minus_acentos_simbolos_num.csv,procesados_all_balan_minus_acentos_simbolos_num_lemma.csv,procesados_all_balan_minus_acentos_simbolos_num_lemma_palVac.csv,procesados_all_balan_minus_acentos_simbolos_num_palVac.csv,procesados_all_balan_minus_acentos_simbolos_palVac.csv,procesados_all_balan_minus_simbolos.csv,procesados_all_balan_minus_simbolos_lemma.csv,procesados_all_balan_minus_simbolos_lemma_palVac.csv,procesados_all_balan_minus_simbolos_num.csv,procesados_all_balan_minus_simbolos_num_lemma.csv,procesados_all_balan_minus_simbolos_num_lemma_palVac.csv,procesados_all_balan_minus_simbolos_num_palVac.csv,procesados_all_balan_minus_simbolos_palVac.csv
```

### Construcción de matriz embeddings

```bash
python .\src\construccion_matriz_embedding.py procesos=construccion_matriz procesos.data_path_input=dataset_codificado_all_balan_minus_acentos_simbolos_num.pkl procesos.data_path_embedding=SBW-vectors-300-min5.txt
```

```bash
python .\src\construccion_matriz_embedding.py procesos=construccion_matriz procesos.data_path_input=dataset_codificado_all_balan_acentos_simbolos_num_lemma.csv procesos.data_path_embedding=SBW-vectors-300-min5.txt
```

### Entrenamiento 

```bash
python .\src\bilstm2.py --multirun procesos=entrenamiento procesos.entrenamiento.path_input=
```


```bash
python .\src\bilstm2.py procesos=entrenamiento 
```

```bash
python .\src\bilstm2.py --multirun procesos=entrenamiento procesos/entrenamiento=bilstm procesos.entrenamiento.EPOCAS=10 procesos.entrenamiento.OPTIMIZADOR=adam,adamw,adamfactor procesos.entrenamiento.ls=0.0004,0.0005,0.0006 procesos.entrenamiento.BATCH_SIZE=32,64 procesos.entrenamiento.HIDDEN_DIM=32,64 procesos.entrenamiento.path_input=
```

```bash
python .\src\bilstm2.py --multirun procesos=entrenamiento procesos/entrenamiento=bilstm procesos.entrenamiento.EPOCAS=10 procesos.entrenamiento.OPTIMIZADOR=adamw procesos.entrenamiento.ls=0.002 procesos.entrenamiento.BATCH_SIZE=128 procesos.entrenamiento.HIDDEN_DIM=128 procesos.entrenamiento.DROPOUT=0.6,0.7 procesos.entrenamiento.TAM_MAXIMO=10 procesos.entrenamiento.path_input=matriz_embeddings_SBW_all_balan_minus_acentos_simbolos_num.pkl
```