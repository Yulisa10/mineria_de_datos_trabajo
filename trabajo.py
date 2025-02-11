import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import gzip
import pickle

# Configurar la aplicación
st.set_page_config(page_title="Análisis de Detección de Ocupación", layout="wide")
st.title("Análisis de Detección de Ocupación")
st.write("Grupo: Yulisa Ortiz Giraldo y Juan Pablo Noreña Londoño")

@st.cache_data
def load_data():
    df_train = pd.read_csv("https://raw.githubusercontent.com/JuanPablo9999/Mineria_de_datos_streamlit/main/datatrain.csv")
    df_test = pd.read_csv("https://raw.githubusercontent.com/JuanPablo9999/Mineria_de_datos_streamlit/main/datatest.csv")
    df = pd.concat([df_train, df_test], axis=0)
    df.drop(columns=["id", "date"], inplace=True, errors='ignore')
    return df

df = load_data()

@st.cache_data
def preprocess_data(df):
    X = df.drop(columns=["Occupancy"], errors='ignore')
    y = df["Occupancy"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

X, y, scaler = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def load_xgb_model():
    with gzip.open('xgb_model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    return model

xgb_model = load_xgb_model()

# Crear la barra lateral
seccion = st.sidebar.radio("Tabla de Contenidos", [
    "Vista previa de los datos", "Análisis Descriptivo", "Mapa de Calor", "Modelo XGBoost", "Predicción"
])

if seccion == "Vista previa de los datos":
    st.subheader("Vista previa de los datos")
    st.write(df.head())
    st.write("### Información del Dataset")
    st.write(df.info())
    st.write("Distribución de la variable objetivo")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Occupancy"], ax=ax)
    st.pyplot(fig)

elif seccion == "Análisis Descriptivo":
    st.subheader("Análisis Descriptivo")
    st.write(df.describe())
    st.write("Histogramas de variables clave")
    cols = ["Temperature", "Humidity", "Light", "CO2"]
    for col in cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], bins=30, kde=True, ax=ax)
        ax.set_title(f"Distribución de {col}")
        st.pyplot(fig)

elif seccion == "Mapa de Calor":
    st.subheader("Mapa de Calor de Correlaciones")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap="coolwarm", annot=True, ax=ax)
    st.pyplot(fig)

elif seccion == "Modelo XGBoost":
    st.subheader("Evaluación del Modelo XGBoost")
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    
    st.write("### Métricas de Evaluación")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**Precision:** {precision:.4f}")

elif seccion == "Predicción":
    st.subheader("Hacer una Predicción con XGBoost")
    input_data = {}
    for col in df.drop(columns=["Occupancy"], errors='ignore').columns:
        input_data[col] = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = xgb_model.predict(input_scaled)
    resultado = "Ocupado" if prediction[0] == 1 else "No Ocupado"
    st.write(f"**Predicción:** {resultado}")
    
    
elif seccion == "Modelo de redes neuronales":
    st.subheader("Modelo planteado con redes neuronales")

 
# Definir el modelo
model = Sequential([
    Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
clf = model.fit(X_train, y_train, epochs=50, batch_size=500, verbose=0, validation_data=(X_test, y_test))

# Graficar la evolución del entrenamiento
plt.figure(figsize=(12, 5))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(clf.history['loss'], label='Entrenamiento')
plt.plot(clf.history['val_loss'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Evolución de la pérdida')
plt.legend()

# Gráfico de precisión
plt.subplot(1, 2, 2)
plt.plot(clf.history['accuracy'], label='Entrenamiento')
plt.plot(clf.history['val_accuracy'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.title('Evolución de la precisión')
plt.legend()

plt.show()

# Predicciones en X_test
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Comparación gráfica de valores reales vs predichos
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, label='Valores Reales', alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label='Predicciones', alpha=0.6)
plt.xlabel('Índice de muestra')
plt.ylabel('Clase')
plt.title('Comparación entre valores reales y predichos')
plt.legend()
plt.show()


