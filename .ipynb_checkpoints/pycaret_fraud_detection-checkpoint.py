import pandas as pd
from pycaret.classification import *
import time

def main():
    """
    Función principal para ejecutar el pipeline de detección de fraude con PyCaret.
    """
    print("Iniciando el proceso de detección de fraude con PyCaret...")

    # --- 1. Carga de Datos ---
    print("Paso 1: Cargando los conjuntos de datos...")
    try:
        df_tran = pd.read_csv('Machine-Learning/train_transaction.csv')
        df_id = pd.read_csv('Machine-Learning/train_identity.csv')
        print("Datos cargados exitosamente.")
    except FileNotFoundError:
        print("\nERROR: Archivos de datos no encontrados.")
        print("Por favor, descarga los datos desde https://www.kaggle.com/c/ieee-fraud-detection/data")
        print("y colócalos en la carpeta 'Machine-Learning/'.")
        return

    # --- 2. Fusión y Preparación de Datos ---
    print("Paso 2: Fusionando y preparando los datos...")
    df = pd.merge(df_tran, df_id, on='TransactionID', how='left')

    # Liberar memoria
    del df_tran, df_id
    
    # Reducir el tamaño del dataset para una ejecución más rápida (opcional, para pruebas)
    # df = df.sample(n=50000, random_state=42)

    print(f"El conjunto de datos combinado tiene {df.shape[0]} filas y {df.shape[1]} columnas.")

    # --- 3. Configuración del Entorno de PyCaret ---
    # PyCaret se encargará de la mayoría del preprocesamiento.
    # Las columnas 'V' son anónimas y muchas tienen valores faltantes, PCA es una buena estrategia.
    v_features = [f'V{i}' for i in range(1, 340)]

    print("Paso 3: Configurando el entorno de PyCaret...")
    print("Esto puede tardar varios minutos, ya que PyCaret procesará los datos...")
    
    # Iniciar el temporizador para el setup
    start_time_setup = time.time()

    clf_setup = setup(
        data=df,
        target='isFraud',
        session_id=42,
        log_experiment=True,
        experiment_name='fraud_detection_v1',
        
        # --- Preprocesamiento ---
        numeric_imputation='mean',
        categorical_imputation='mode',
        
        # --- Ingeniería de Características ---
        pca=True,
        pca_method='linear',
        pca_components=30, # Similar al notebook
        ignore_features=['TransactionID'], # Ignorar IDs
        
        # --- Manejo de Desequilibrio ---
        fix_imbalance=True,
        fix_imbalance_method='SMOTE', # Mismo método que en el notebook
        
        # --- Eficiencia ---
        n_jobs=-1, # Usar todos los cores de la CPU
        use_gpu=False, # Cambiar a True si se tiene una GPU compatible
        
        silent=True, # Para reducir la verbosidad de la salida
        verbose=False
    )
    
    end_time_setup = time.time()
    print(f"Entorno de PyCaret configurado en {((end_time_setup - start_time_setup) / 60):.2f} minutos.")

    # --- 4. Comparación de Modelos ---
    print("\nPaso 4: Comparando modelos para encontrar el mejor...")
    print("PyCaret entrenará y evaluará varios modelos automáticamente.")
    
    # Iniciar el temporizador para la comparación
    start_time_compare = time.time()

    # Comparamos modelos y ordenamos por AUC, ya que es una buena métrica para clases desequilibradas.
    best_model = compare_models(sort='AUC', n_select=1, fold=3) # Usamos 3 folds para agilizar
    
    end_time_compare = time.time()
    print(f"Comparación de modelos completada en {((end_time_compare - start_time_compare) / 60):.2f} minutos.")
    print("\nLeaderboard de Modelos (Top 5):")
    print(pull().head(5))


    # --- 5. Creación y Evaluación del Modelo Final ---
    print("\nPaso 5: Creando y evaluando el modelo final...")
    
    # El mejor modelo ya está seleccionado por compare_models
    final_model = best_model
    
    print(f"El mejor modelo seleccionado es: {type(final_model).__name__}")

    # No es necesario hacer tune_model() para esta demostración, 
    # ya que compare_models da una excelente línea base.
    
    print("Generando gráficos de evaluación del modelo...")
    
    # Guardar gráficos en archivos
    plot_model(final_model, plot='auc', save=True)
    print("- Gráfico AUC guardado en 'AUC.png'")
    
    plot_model(final_model, plot='confusion_matrix', save=True)
    print("- Matriz de Confusión guardada en 'Confusion Matrix.png'")
    
    plot_model(final_model, plot='feature', save=True)
    print("- Gráfico de Importancia de Características guardado en 'Feature Importance.png'")
    
    # Evaluar en el conjunto de prueba (hold-out)
    hold_out_results = predict_model(final_model)
    print("\nResultados del modelo en el conjunto de prueba (Hold-out):")
    print(pull())


    # --- 6. Guardar el Pipeline ---
    print("\nPaso 6: Guardando el pipeline del modelo final...")
    save_model(final_model, 'final_fraud_detection_pipeline')
    print("Pipeline guardado como 'final_fraud_detection_pipeline.pkl'")

    print("\n¡Proceso completado exitosamente!")
    print("Puedes encontrar el modelo guardado y los gráficos de evaluación en el directorio actual.")


if __name__ == "__main__":
    # Instalar PyCaret si no está presente
    try:
        import pycaret
    except ImportError:
        print("PyCaret no está instalado. Instalando ahora...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pycaret[full]"])
    
    main()
