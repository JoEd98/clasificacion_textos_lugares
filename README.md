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

```bash
python .\src\pre-procesamiento.py procesos=preprocesamiento procesos/preprocesamiento=simple
```

Obtener columna de texto y columna clase sin agregar datos y con balancear clases

```bash
python .\src\pre-procesamiento.py procesos=preprocesamiento procesos/preprocesamiento=undersampling procesos.preprocesamiento.path_data_input=./data/reparados/reparados_all.csv
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
python src/procesamiento.py procesos=procesamiento procesos/procesamiento=simple procesos.procesamiento.quitar_ascentos=true procesos.procesamiento.path_data_input=./data/pre-procesados/preprocesados_30_balan.csv
```
### Caracterización


