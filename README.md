#  Telecom Customer Churn Predictor

Este proyecto implementa una solución de Machine Learning de extremo a extremo para predecir la fuga de clientes (*churn*) en una empresa de telecomunicaciones. El modelo final permite identificar usuarios en riesgo antes de que cancelen su suscripción, facilitando estrategias de retención proactivas.

---

##  Estructura del Proyecto

```text
├── data/                  # Conjunto de datos original
├── notebooks/             # Cuadernos de Jupyter para experimentación
│   ├── 01_exploracion_datos.ipynb
│   ├── modelo_churn_arbol.pkl
│   └── columnas_modelo.pkl
├── predict.py             # Script ejecutable para entornos de producción
└── README.md              # Documentación del proyecto
```

---

##  Resumen del Desarrollo

### 1. Análisis y Limpieza de Datos
* **Insight de Negocio:** Se descubrió visualmente que los clientes con contratos mes a mes (*Month-to-month*) presentan tasas de abandono significativamente mayores que aquellos con contratos anuales.
* **Ingeniería de Datos:** Se corrigieron valores nulos ocultos (espacios en blanco) en la columna `TotalCharges` y se transformaron las variables categóricas mediante técnicas de *One-Hot Encoding*.

### 2. Modelado e Inteligencia Artificial
* **Algoritmo:** Se implementó un modelo basado en un **Árbol de Decisión** (`DecisionTreeClassifier`).
* **Métricas:** Tras optimizar e incluir variables de contexto (tipo de internet, método de pago, etc.), el modelo alcanzó una **precisión general del 78.39%**.
* **Feature Importance:** La IA demostró fijarse principalmente en la antigüedad del cliente (`tenure`) y los cargos mensuales (`MonthlyCharges`) para tomar decisiones predictivas.

---

##  Instalación y Uso

Sigue estos pasos para clonar el repositorio y ejecutar el predictor en tu entorno local.

### Prerrequisitos
Asegúrate de contar con Python 3.10 o superior instalado.

### 1. Clonar el repositorio
```bash
git clone https://github.com
cd customer-churn-analytics
```

### 2. Instalar las dependencias necesarias
```bash
pip install pandas scikit-learn joblib
```

### 3. Ejecutar el script de producción
Para evaluar un caso de prueba y obtener una predicción en tiempo real desde la terminal, ejecuta:
```bash
python predict.py
```

---

##  Impacto en el Negocio
El script de producción está diseñado para integrarse con sistemas CRM o herramientas de automatización de marketing. Al detectar clientes con altas probabilidades de fuga (ej. **71.20% de riesgo**), el sistema puede disparar alertas automáticas para ofrecer incentivos de retención personalizados, optimizando el presupuesto de marketing y protegiendo los ingresos de la compañía.
