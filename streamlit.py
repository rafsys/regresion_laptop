import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuración de página
st.set_page_config(
    page_title="Predicción de Precios de Laptops", layout="wide"
)

# ---------------------------------------------------------
# CSS Personalizado
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Estilo general del contenedor principal */
    .main {
        background-color: #f8f9fa;
    }
    /* Estilo de la barra lateral */
    [data-testid="stSidebar"] {
        background-color: #1e1e2f;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Tarjetas de métricas */
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Carga y Preparación de Datos (Cacheado)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("laptop_data.csv")  # Asegurar ruta correcta
    return df


@st.cache_resource
def train_model(df):
    X = df.drop(columns=["Price"])
    y = df["Price"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predicciones de prueba
    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }

    return (
        model,
        scaler,
        X_encoded.columns,
        metrics,
        y_test,
        y_pred,
        num_cols,
        cat_cols,
    )


# Cargar datos e inicializar
try:
    df = load_data()
    (
        model,
        scaler,
        feature_names,
        metrics,
        y_test,
        y_pred,
        num_cols,
        cat_cols,
    ) = train_model(df)
except Exception as e:
    st.error(
        f"Error al cargar el archivo CSV o entrenar el modelo. Verifica que 'laptop_data.csv' esté en la carpeta raíz. Detalle: {e}"
    )
    st.stop()

# ---------------------------------------------------------
# BARRA LATERAL (Sidebar)
# ---------------------------------------------------------
st.sidebar.image(
    "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?auto=format&fit=crop&w=400&q=80",
    caption="Laptop Price Predictor",
    use_container_width=True,
)

st.sidebar.title("Navegación")
opcion = st.sidebar.radio(
    "Seleccione una sección:",
    [
        "Descripción del Modelo",
        "Visualizar Dataset",
        "Métricas del Modelo",
        "Predicción con Datos Nuevos",
    ],
)

# ---------------------------------------------------------
# SECCIÓN 1: DESCRIPCIÓN DEL MODELO
# ---------------------------------------------------------
if opcion == "Descripción del Modelo":
    st.title("💻 Descripción del Modelo y Contexto")

    st.markdown(
        """
    ### Contexto del Proyecto
    El mercado de computadoras portátiles es altamente dinámico y variable. Este modelo busca estimar de forma precisa el **precio comercial** de una laptop a partir de sus especificaciones técnicas y características de hardware.
    
    ### Algoritmo Utilizado
    * **Tipo de Aprendizaje:** Supervisado.
    * **Algoritmo:** **Random Forest Regressor** (Bosques Aleatorios de Regresión).
    * **¿Por qué este modelo?:** Es ideal para manejar relaciones no lineales complejas entre características técnicas y el precio, combina múltiples árboles de decisión para minimizar el sobreajuste y gestiona eficientemente datos mixtos (numéricos y categóricos).
    """
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔢 Variables Numéricas (Features)")
        for col in num_cols:
            st.write(f"- **{col}**: Min: `{df[col].min()}`, Max: `{df[col].max()}`")

    with col2:
        st.subheader("🏷️ Variables Categóricas (Features)")
        for col in cat_cols:
            st.write(
                f"- **{col}**: `{df[col].nunique()}` categorías únicas (Ej: {', '.join(map(str, df[col].unique()[:3]))}...)"
            )

# ---------------------------------------------------------
# SECCIÓN 2: DATASET
# ---------------------------------------------------------
elif opcion == "Visualizar Dataset":
    st.title("📊 Dataset de Laptops")
    st.dataframe(df, use_container_width=True)

    st.subheader("Estadísticas Descriptivas")
    st.write(df.describe())

# ---------------------------------------------------------
# SECCIÓN 3: MÉTRICAS
# ---------------------------------------------------------
elif opcion == "Métricas del Modelo":
    st.title("📈 Métricas de Rendimiento")

    col1, col2, col3 = st.columns(3)
    col1.metric("Coeficiente de Determinación (R²)", f"{metrics['r2']:.4f}")
    col2.metric("Error Absoluto Medio (MAE)", f"${metrics['mae']:.2f}")
    col3.metric("Raíz del Error Cuadrático (RMSE)", f"${metrics['rmse']:.2f}")

    st.markdown("---")
    st.subheader("Evaluación Gráfica: Valores Reales vs. Predichos")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color="#1f77b4", ax=ax)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "--r",
        linewidth=2,
    )
    ax.set_xlabel("Precio Real")
    ax.set_ylabel("Precio Predicho")
    ax.set_title("Ajuste de Regresión (Línea roja = Ajuste Perfecto)")
    st.pyplot(fig)

    st.info(
        "💡 **Nota sobre la Matriz de Confusión:** La matriz de confusión solo aplica para algoritmos de **clasificación**. Al tratarse de un problema de **regresión** (predicción de un valor continuo como el precio), las métricas adecuadas son R², MAE y RMSE, complementadas con el gráfico de dispersión de errores."
    )

# ---------------------------------------------------------
# SECCIÓN 4: PREDICCIÓN CON DATOS NUEVOS
# ---------------------------------------------------------
elif opcion == "Predicción con Datos Nuevos":
    st.title("🔮 Formulario de Predicción")
    st.write(
        "Ingrese las características de la laptop para predecir su precio exacto:"
    )

    inputs = {}
    with st.form("prediction_form"):
        # Controles dinámicos restringidos a valores válidos del dataset
        for col in df.drop(columns=["Price"]).columns:
            if col in num_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                inputs[col] = st.number_input(
                    f"{col}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                )
            elif col in cat_cols:
                options = df[col].unique().tolist()
                inputs[col] = st.selectbox(f"{col}:", options=options)

        submit = st.form_submit_button("Calcular Precio Estimado")

    if submit:
        # Formatear la entrada del usuario al formato entrenado
        input_df = pd.DataFrame([inputs])
        input_encoded = pd.get_dummies(input_df)

        # Reindexar para alinearse exactamente con las columnas codificadas del entrenamiento
        input_encoded = input_encoded.reindex(
            columns=feature_names, fill_value=0
        )
        input_scaled = scaler.transform(input_encoded)

        # Predicción
        pred_price = model.predict(input_scaled)[0]

        st.success(f"💰 **Precio Estimado:** ${pred_price:,.2f}")
