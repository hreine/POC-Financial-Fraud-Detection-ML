# Detección de Fraude en Transacciones Financieras con IA Explicable (XAI)

<p align="center" width="100%">
<img alt="GIF" src="https://user-images.githubusercontent.com/31254745/191377492-9b827999-aba9-4dc7-8adf-fdb1b6c8fb19.png">
</p>

Este proyecto es una Prueba de Concepto (PoC) que demuestra cómo construir un sistema de detección de fraude de extremo a extremo utilizando técnicas de Machine Learning y, lo que es más importante, cómo hacer que sus decisiones sean transparentes y comprensibles utilizando la IA Explicable (XAI).

El repositorio te guiará a través de un cuaderno de Jupyter (`1_Project_EDA.ipynb`) que cubre todo el ciclo de vida de un proyecto de ciencia de datos, desde el análisis exploratorio de datos hasta el despliegue de un modelo de clasificación robusto.

## 📜 Tabla de Contenidos
- [Introducción al Problema](#-introducción-al-problema)
- [Objetivos del Proyecto](#-objetivos-del-proyecto)
- [🚀 Cómo Empezar](#-cómo-empezar)
  - [Prerrequisitos](#prerrequisitos)
  - [Instalación](#instalación)
- [📝 Tutorial del Proyecto: Paso a Paso](#-tutorial-del-proyecto-paso-a-paso)
  - [Paso 1: Análisis Exploratorio de Datos (EDA)](#paso-1-análisis-exploratorio-de-datos-eda)
  - [Paso 2: Ingeniería de Características (Feature Engineering)](#paso-2-ingeniería-de-características-feature-engineering)
  - [Paso 3: Preprocesamiento de Datos](#paso-3-preprocesamiento-de-datos)
  - [Paso 4: Entrenamiento y Selección del Modelo](#paso-4-entrenamiento-y-selección-del-modelo)
  - [Paso 5: Evaluación del Modelo](#paso-5-evaluación-del-modelo)
- [🧠 IA Explicable (XAI): Entendiendo las Predicciones](#-ia-explicable-xai-entendiendo-las-predicciones)
- [🏆 Conclusión y Resultados Clave](#-conclusión-y-resultados-clave)
- [💡 Futuras Mejoras](#-futuras-mejoras)

## 🏦 Introducción al Problema

El fraude en transacciones financieras es un problema masivo y creciente. Con el auge de los pagos digitales, los estafadores desarrollan constantemente nuevos métodos para realizar transacciones fraudulentas, causando pérdidas millonarias a consumidores y empresas.

Los modelos de Machine Learning son increíblemente efectivos para detectar estos patrones de fraude, pero a menudo funcionan como una "caja negra". Un analista de fraude o un gerente de negocio no puede simplemente confiar en una predicción de "fraude" sin entender *por qué* el modelo tomó esa decisión. ¿Fue por la ubicación inusual? ¿El monto de la transacción? ¿La hora del día?

Aquí es donde entra en juego la **IA Explicable (XAI)**. XAI nos proporciona las herramientas para abrir esa caja negra y entender los factores que impulsan las predicciones del modelo, generando confianza y permitiendo una mejor toma de decisiones.

## 🎯 Objetivos del Proyecto

1.  **Construir un Clasificador Robusto:** Desarrollar y comparar varios modelos de Machine Learning para clasificar con precisión las transacciones como fraudulentas o legítimas.
2.  **Evaluar el Rendimiento:** Medir la eficacia de los modelos utilizando métricas clave como ROC AUC, Recall y Precisión, que son cruciales en problemas de clasificación desequilibrada.
3.  **Implementar IA Explicable:** Utilizar técnicas de XAI para interpretar las predicciones del modelo con mejor rendimiento, identificando los factores más influyentes en la detección de fraude.
4.  **Crear una Guía Práctica:** Presentar todo el proceso en un formato de tutorial claro y reproducible.

## 🚀 Cómo Empezar

Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local.

### Prerrequisitos

- Python 3.7 o superior
- Git

### Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu_usuario/tu_repositorio.git
    cd POC-Financial-Fraud-Detection-ML
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

3.  **Instala las dependencias:**
    Hemos incluido un archivo `requirements.txt` para facilitar la instalación.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descarga los datos:**
    Debido a su tamaño, los datos del concurso de Kaggle no están incluidos en este repositorio. Debes descargarlos desde la [página de la competición de Detección de Fraude de IEEE-CIS](https://www.kaggle.com/c/ieee-fraud-detection/data) y colocarlos en la carpeta `Machine-Learning/`. Necesitarás los siguientes archivos:
    - `train_transaction.csv`
    - `train_identity.csv`

5.  **Inicia Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Esto abrirá una pestaña en tu navegador. Navega a la carpeta `Machine-Learning` y abre el archivo `1_Project_EDA.ipynb`.

## 📝 Tutorial del Proyecto: Paso a Paso

El cuaderno de Jupyter es la pieza central de este proyecto. A continuación, se resume el flujo de trabajo que encontrarás.

### Paso 1: Análisis Exploratorio de Datos (EDA)

Comenzamos con un análisis profundo de los datos para entender su estructura, identificar valores faltantes y descubrir patrones iniciales.
- **Desequilibrio de Clases:** El primer hallazgo crucial es que el conjunto de datos está **altamente desequilibrado**. Solo un 3.5% de las transacciones son fraudulentas. Esto tiene implicaciones importantes para el entrenamiento y la evaluación del modelo.
- **Visualización de Características:** Analizamos la distribución de variables clave como `TransactionAmt` (monto de la transacción) y `ProductCD` para ver cómo difieren entre transacciones fraudulentas y legítimas.

### Paso 2: Ingeniería de Características (Feature Engineering)

Creamos nuevas características para ayudar al modelo a capturar mejor los patrones de fraude.
- **Características Temporales:** Extraemos la hora del día y el día de la semana de la característica `TransactionDT`.
- **Agrupación de Dominios de Email:** Los dominios de correo electrónico se limpian y agrupan en categorías más generales (ej., `gmail.com`, `yahoo.com`, `otros`).
- **Interacciones de Características:** Creamos nuevas características combinando `card1`, `addr1` y otras para capturar patrones más complejos.

### Paso 3: Preprocesamiento de Datos

Preparamos los datos para el entrenamiento del modelo.
- **Manejo de Valores Faltantes:** Rellenamos los valores faltantes utilizando estrategias apropiadas para cada tipo de característica.
- **Codificación de Variables Categóricas:** Convertimos las características categóricas en representaciones numéricas usando `LabelEncoder`.
- **Reducción de Dimensionalidad (PCA):** El conjunto de datos contiene más de 300 características anónimas (`V1`-`V339`). Usamos **Análisis de Componentes Principales (PCA)** para reducir estas características a 30 componentes principales, reteniendo la mayor parte de la varianza mientras reducimos la complejidad del modelo.
- **Manejo del Desequilibrio de Clases (SMOTE):** Para abordar el desequilibrio de clases, aplicamos la técnica **SMOTE (Synthetic Minority Over-sampling Technique)**. SMOTE crea ejemplos sintéticos de la clase minoritaria (fraude) en el conjunto de entrenamiento, ayudando al modelo a aprender mejor sus características sin simplemente predecir la clase mayoritaria.

### Paso 4: Entrenamiento y Selección del Modelo

Entrenamos y comparamos varios modelos de clasificación:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- **LightGBM (LGBM)**

El modelo **LightGBM** demostró ser el de mejor rendimiento, logrando un excelente equilibrio entre velocidad y precisión.

### Paso 5: Evaluación del Modelo

Evaluamos el modelo LightGBM en un conjunto de validación que no se utilizó durante el entrenamiento.
- **Métricas Clave:**
  - **ROC AUC:** 0.931. Una puntuación excelente que indica una alta capacidad para distinguir entre clases.
  - **Recall:** 0.728. Esto significa que el modelo identificó correctamente casi el 73% de todas las transacciones fraudulentas.
  - **Matriz de Confusión:** Proporciona un desglose detallado de los verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.

## 🧠 IA Explicable (XAI): Entendiendo las Predicciones

Utilizamos la propiedad `feature_importance_` del modelo LightGBM para entender qué características fueron más importantes para sus decisiones.
- **Visualización de Importancia:** Se genera un gráfico de barras que muestra las 20 características más influyentes.
- **Principales Conclusiones de XAI:** Características como `TransactionDT`, `TransactionAmt`, `card1`, y varias de las componentes principales de PCA resultaron ser determinantes clave para predecir el fraude. Esta información es invaluable para un analista, ya que valida que el modelo está "pensando" de una manera lógica y centrada en los datos correctos.

## 🏆 Conclusión y Resultados Clave

Este proyecto demuestra con éxito la construcción de un pipeline de detección de fraude de alto rendimiento.
- **Mejor Modelo:** LightGBM.
- **Rendimiento Clave:** **ROC AUC de 0.931** y **Recall de 0.728** en el conjunto de validación.
- **Explicabilidad:** Demostramos que es posible y necesario abrir la "caja negra" de los modelos de Machine Learning para generar confianza y proporcionar información procesable a los expertos en el dominio.

## 💡 Futuras Mejoras

- **Modelos más Avanzados:** Explorar arquitecturas de Redes Neuronales Profundas (Deep Learning) para capturar patrones aún más sutiles.
- **Técnicas XAI Adicionales:** Implementar SHAP (SHapley Additive exPlanations) para obtener explicaciones a nivel de transacción individual.
- **Despliegue de una API:** Envolver el modelo en una API REST para que pueda ser consumido por otras aplicaciones y realizar predicciones en tiempo real.
- **Aplicación Web Interactiva:** Desarrollar una interfaz de usuario donde un analista pueda ingresar los datos de una transacción y recibir no solo una predicción, sino también una explicación visual de por qué se tomó esa decisión.