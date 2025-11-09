# POC de Detección de Fraude con Machine Learning y IA Explicable (XAI)

<p align="center" width="100%">
  <img alt="Banner del Proyecto" src="https://user-images.githubusercontent.com/31254745/191377492-9b827999-aba9-4dc7-8adf-fdb1b6c8fb19.png" width="600">
</p>

<p align="center">
  <strong>Un proyecto de extremo a extremo para identificar transacciones financieras fraudulentas utilizando Python, PyCaret y MLflow, con un enfoque en la interpretabilidad del modelo.</strong>
</p>

---

## 📜 Visión General del Proyecto

Este repositorio presenta una Prueba de Concepto (PoC) completa para un sistema de detección de fraude en transacciones financieras. El objetivo principal no es solo construir un modelo de Machine Learning de alta precisión, sino también demostrar cómo podemos **interpretar y confiar en sus decisiones** a través de la Inteligencia Artificial Explicable (XAI).

En la industria financiera, un modelo de "caja negra" no es suficiente. Los reguladores, analistas y clientes necesitan entender *por qué* una transacción se marca como fraudulenta. Este proyecto aborda esa necesidad de frente.

### Principales Tecnologías Utilizadas
- **Lenguaje:** Python
- **Análisis y Modelado:** Pandas, Scikit-learn, Jupyter
- **AutoML:** PyCaret
- **Modelos Avanzados:** LightGBM, XGBoost
- **Seguimiento de Experimentos:** MLflow
- **Visualización:** Matplotlib, Seaborn

---

## 🎯 Objetivos

1.  **Análisis Exploratorio de Datos (EDA):** Comprender a fondo un conjunto de datos transaccionales complejo y altamente desequilibrado.
2.  **Ingeniería de Características:** Crear y transformar características para mejorar el poder predictivo del modelo.
3.  **Modelado y Comparación:** Entrenar, comparar y optimizar múltiples algoritmos de clasificación, desde regresión logística hasta Gradient Boosting.
4.  **Automatización con PyCaret:** Demostrar cómo una librería de AutoML como PyCaret puede acelerar drásticamente el ciclo de vida del modelado.
5.  **Experiment Tracking con MLflow:** Registrar y visualizar sistemáticamente todos los experimentos, parámetros y resultados para una reproducibilidad total.
6.  **IA Explicable (XAI):** "Abrir la caja negra" para identificar los factores clave que utilizan nuestros modelos para detectar fraudes.

---

## 📂 Estructura del Repositorio

```
POC-Financial-Fraud-Detection-ML/
│
├── Machine-Learning/
│   ├── 1_Project_EDA.ipynb             # Cuaderno principal con análisis manual detallado.
│   ├── 1_Project_EDA_pycaret.ipynb     # Cuaderno alternativo usando AutoML con PyCaret.
│   ├── final_fraud_detection_pipeline.pkl # Pipeline del modelo final entrenado y listo para usar.
│   └── mlruns/                         # Directorio de datos de MLflow (experimentos, modelos).
│
├── .gitignore
├── README.md                           # Este archivo.
└── requirements.txt                    # Dependencias del proyecto.
```

---

## 🚀 Cómo Empezar: Guía de Instalación

Sigue estos pasos para tener el proyecto funcionando en tu máquina local.

### 1. Prerrequisitos
- Python 3.7+
- Git

### 2. Clonar y Configurar el Entorno

```bash
# Clona el repositorio
git clone https://github.com/tu_usuario/POC-Financial-Fraud-Detection-ML.git
cd POC-Financial-Fraud-Detection-ML

# (Recomendado) Crea y activa un entorno virtual
python -m venv venv
# En macOS/Linux:
source venv/bin/activate
# En Windows:
.\venv\Scripts\activate
```

### 3. Instalar Dependencias

Instala todas las librerías necesarias usando el archivo `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Descargar el Conjunto de Datos

Este proyecto utiliza el conjunto de datos de la competición [IEEE-CIS Fraud Detection en Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data).

1.  Ve a la página de Kaggle y descarga los siguientes archivos:
    - `train_transaction.csv`
    - `train_identity.csv`
    - `test_transaction.csv`
    - `test_identity.csv`
2.  Crea una carpeta `data` dentro de `Machine-Learning`.
3.  Mueve los archivos descargados a la nueva carpeta: `Machine-Learning/data/`.

### 5. Iniciar el Servidor de Jupyter

```bash
jupyter notebook
```

Abre uno de los cuadernos en la carpeta `Machine-Learning`:
- `1_Project_EDA.ipynb`: Para un recorrido manual y detallado.
- `1_Project_EDA_pycaret.ipynb`: Para ver el poder de la automatización.

### 6. (Opcional) Visualizar los Experimentos con MLflow

Este proyecto está integrado con MLflow para rastrear cada experimento. Para iniciar la interfaz de usuario de MLflow y ver los resultados, ejecuta:

```bash
# Asegúrate de estar en el directorio raíz del proyecto
mlflow ui --backend-store-uri Machine-Learning/mlruns
```

Abre tu navegador en `http://localhost:5000` para explorar los parámetros, métricas y artefactos de cada ejecución del modelo.

---

## 💡 Creación de una Aplicación PoC con Streamlit

El pipeline final, `final_fraud_detection_pipeline.pkl`, contiene todo el proceso de preprocesamiento y el modelo entrenado. Puedes usarlo para construir fácilmente una aplicación interactiva con [Streamlit](https://streamlit.io/).

1.  **Instala Streamlit:** `pip install streamlit`
2.  **Crea un archivo `app.py`:**

```python
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Cargar el pipeline de detección de fraude entrenado
pipeline = load_model('Machine-Learning/final_fraud_detection_pipeline')

st.title("PoC: Detección de Fraude Financiero")
st.write("Esta aplicación utiliza un modelo de Machine Learning para predecir si una transacción es fraudulenta.")

# Crear campos de entrada para las características más importantes
# (Estos son ejemplos, se pueden añadir más según el modelo)
transaction_amt = st.number_input("Monto de la Transacción (TransactionAmt)", min_value=0.0, format="%.2f")
product_cd = st.selectbox("Código del Producto (ProductCD)", ['W', 'C', 'H', 'S', 'R'])
card1 = st.number_input("Valor de 'card1'", min_value=0)
addr1 = st.number_input("Valor de 'addr1'", min_value=0)
# Añade más campos según las características de tu modelo

# Crear un DataFrame con los datos de entrada
input_data = pd.DataFrame({
    'TransactionAmt': [transaction_amt],
    'ProductCD': [product_cd],
    'card1': [card1],
    'addr1': [addr1],
    # Asegúrate de incluir todas las columnas que el modelo espera.
    # Puedes rellenar las demás con valores por defecto o solicitar más entradas.
})

if st.button("Predecir"):
    # Realizar la predicción
    predictions = predict_model(pipeline, data=input_data)
    
    # Extraer el resultado
    is_fraud = predictions['prediction_label'].iloc[0]
    score = predictions['prediction_score'].iloc[0]
    
    if is_fraud == 1:
        st.error(f"¡Transacción Marcada como Fraude! (Confianza: {score:.2f})")
    else:
        st.success(f"Transacción Legítima (Confianza: {score:.2f})")
        
    st.write("Detalles de la Predicción:")
    st.write(predictions)

```

3.  **Ejecuta la aplicación:**

```bash
streamlit run app.py
```
Esta aplicación sirve como un ejemplo práctico de cómo el modelo puede ser desplegado para que un analista de fraude lo utilice en su día a día.
