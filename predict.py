import joblib
import pandas as pd

print("🔄 Cargando el modelo de Inteligencia Artificial...")
# 1. Cargar el cerebro de la IA y sus columnas guardadas desde la carpeta correcta
modelo = joblib.load('notebooks/modelo_churn_arbol.pkl')
columnas_modelo = joblib.load('notebooks/columnas_modelo.pkl')

# 2. SIMULACIÓN: Datos de un cliente nuevo en riesgo
# Es un cliente con contrato mes a mes, fibra óptica y alta facturación
cliente_nuevo = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 2,                     # Lleva solo 2 meses
    'InternetService': 'Fiber optic', # Servicio caro
    'Contract': 'Month-to-month',     # Contrato inestable
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 85.50,          # Pago mensual alto
    'TotalCharges': 171.00
}

print("📊 Procesando los datos del cliente...")
# 3. Convertir el cliente a una tabla de Pandas
df_cliente = pd.DataFrame([cliente_nuevo])

# 4. Aplicar el mismo One-Hot Encoding que hicimos al entrenar
df_cliente_enc = pd.get_dummies(df_cliente)

# 5. Alinear las columnas para que coincidan exactamente con la IA
df_cliente_final = df_cliente_enc.reindex(columns=columnas_modelo, fill_value=0)

# 6. ¡PREDECIR EL FUTURO!
prediccion = modelo.predict(df_cliente_final)[0]
probabilidad = modelo.predict_proba(df_cliente_final)[0][1]

print("\n================ RESULTADO ================")
if prediccion == 1:
    print(f"🚨 ALERTA: Existe un {(probabilidad * 100):.2f}% de probabilidad de que este cliente CANCELE el servicio.")
    print("💡 Acción recomendada: Ofrecer un descuento para moverlo a contrato anual de inmediato.")
else:
    print(f"✅ CLIENTE SEGURO: Hay un {(probabilidad * 100):.2f}% de riesgo de fuga. Mantener campaña estándar.")
print("===========================================\n")
