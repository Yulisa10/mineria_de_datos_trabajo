import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import gzip
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier

# Función para cargar los datos
def load_data():
    url_train = "https://raw.githubusercontent.com/JuanPablo9999/Mineria_de_datos_streamlit/main/datatrain.csv"
    url_test = "https://raw.githubusercontent.com/JuanPablo9999/Mineria_de_datos_streamlit/main/datatest.csv"
    df_train = pd.read_csv(url_train)
    df_test = pd.read_csv(url_test)
    df = pd.concat([df_train, df_test], axis=0)
    df.drop(columns=["id", "date"], inplace=True, errors='ignore')
    return df

df = load_data()

# Preprocesamiento de datos
def preprocess_data(df):
    X = df.drop(columns=["Occupancy"], errors='ignore')
    y = df["Occupancy"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

X, y, scaler = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Función para cargar modelos
def load_pickle_model(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

# Cargar modelos
xgb_model = load_pickle_model('xgb_model.pkl.gz')
nn_model = load_pickle_model('best_model.pkl.gz')

# Interfaz de Streamlit
st.title("Análisis de Detección de Ocupación")
st.sidebar.title("Tabla de Contenidos")
seccion = st.sidebar.radio("Seleccione una sección", [
    "Vista previa de los datos",
    "Información del dataset",
    "Análisis Descriptivo",
    "Mapa de calor de correlaciones",
    "Distribución de la variable objetivo",
    "Boxplots",
    "Modelo XGBoost",
    "Modelo de redes neuronales",
    "Conclusión: Selección del Mejor Modelo"
])

# Mostrar contenido basado en la selección
if seccion == "Vista previa de los datos":
    st.subheader("Vista previa de los datos")
    st.write(df.head())

elif seccion == "Información del dataset":
    st.subheader("Información del dataset")
    st.write(df.info())
    st.write("La base de datos fue obtenida de Kaggle y trata sobre la ocupación de habitaciones...")

elif seccion == "Análisis Descriptivo":
    st.subheader("Resumen de los datos")
    st.write(df.describe())
    fig, ax = plt.subplots()
    sns.histplot(df["Temperature"], bins=30, kde=True, ax=ax)
    ax.set_title("Histograma de Temperature")
    st.pyplot(fig)

elif seccion == "Distribución de la variable objetivo":
    st.subheader("Distribución de la variable objetivo")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Occupancy"], ax=ax)
    st.pyplot(fig)

elif seccion == "Mapa de calor de correlaciones":
    st.subheader("Mapa de calor de correlaciones")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif seccion == "Boxplots":
    st.subheader("Conjunto de boxplots")
    st.image("Boxplots.jpg", use_container_width=True)

elif seccion == "Modelo XGBoost":
    st.subheader("Evaluación del Modelo XGBoost")
    y_pred = xgb_model.predict(X_test)
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.4f}")

elif seccion == "Modelo de redes neuronales":
    st.subheader("Evaluación del Modelo de Redes Neuronales")
    _, test_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)
    st.write(f'Accuracy: {test_accuracy:.4f}')

elif seccion == "Conclusión: Selección del Mejor Modelo":
    st.subheader("Conclusión")
    st.write("El modelo XGBoost fue seleccionado por su mayor precisión y robustez...")
