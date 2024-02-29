# House Price Prediction
### Arquitectura de Productos de Datos 2024
### ITAM

### Descripción 
Este proyecto utiliza datos históricos de precios de casas para predecir el precio de una casa dadas ciertas características. El proyecto está estructurado en tres partes principales: preparación de datos, entrenamiento del modelo y inferencia.

### Estructura del Proyecto 
El repositorio está organizado en las siguientes carpetas y archivos:

* data/: Contiene los subdirectorios raw/ para los datos brutos, prep/ para los datos preparados, inference/ para los datos de inferencia y predictions/ para las predicciones generadas por el modelo. 
* src/: Contiene los scripts de Python para la preparación de datos (prep.py), entrenamiento del modelo (train.py) y la inferencia (inference.py).

### Configuración 
#### Requisitos 
Antes de comenzar, asegúrate de tener Python instalado en tu sistema. Este proyecto también requiere las siguientes bibliotecas de Python: 
* numpy 
* pandas 
* scikit-learn 
* joblib

#### Preparación de Datos 
Para preparar los datos para el entrenamiento y la inferencia, ejecuta el siguiente comando: 

```bash
python src/prep.py 
```

Esto procesará los datos en data/raw/ y los guardará en data/prep/.

#### Entrenamiento del Modelo 
Para entrenar el modelo con los datos preparados, ejecuta: 
```bash
python src/train.py 
```
Esto generará un modelo entrenado y lo guardará como model.joblib en la carpeta del proyecto.

#### Inferencia
Para realizar predicciones con un conjunto de datos de inferencia utilizando el modelo entrenado, ejecuta: 
```bash
python src/inference.py
```
Las predicciones se guardarán en data/predictions/.