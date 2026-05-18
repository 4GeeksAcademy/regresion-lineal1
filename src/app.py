from utils import db_connect
engine = db_connect()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

archivo = "https://breathecode.herokuapp.com/asset/internal-link?id=929&path=medical_insurance_cost.csv"

df = pd.read_csv(archivo)

print("--- Primeras 5 filas ---")
print(df.head())

print("\n--- Tamaño del dataset (Filas, Columnas) ---")
print(df.shape)

print("\n--- Tipos de datos y nulos ---")
df.info()


print("--- Resumen Estadístico ---")
display(df.describe()) 

duplicados = df.duplicated().sum()
print(f"\n--- Número de filas duplicadas: {duplicados} ---")

if duplicados > 0:
    df = df.drop_duplicates()
    print("¡Filas duplicadas eliminadas exitosamente!")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['charges'], kde=True, color='teal')
plt.title('Distribución de los Costos Médicos')
plt.xlabel('Costo ($)')
plt.ylabel('Frecuencia')

plt.subplot(1, 2, 2)
sns.boxplot(x='smoker', y='charges', data=df, palette='Set2', hue='smoker', legend=False)
plt.title('Costo Médico: Fumadores vs No Fumadores')
plt.xlabel('¿Es fumador?')
plt.ylabel('Costo ($)')

plt.tight_layout()
plt.show()

df_procesado = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

print("--- Dataset Procesado (Solo Números) ---")
display(df_procesado.head())

plt.figure(figsize=(10, 6))
sns.heatmap(df_procesado.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Correlación: ¿Qué influye más en el costo?')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X = df_procesado.drop('charges', axis=1)

y = df_procesado['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Datos para entrenar: {X_train.shape[0]} filas")
print(f"Datos para probar: {X_test.shape[0]} filas")

modelo_regresion = LinearRegression()

modelo_regresion.fit(X_train, y_train)
print("\n¡El modelo ha sido entrenado con éxito!")

predicciones = modelo_regresion.predict(X_test)

error_cuadratico_medio = mean_squared_error(y_test, predicciones)
r_cuadrado = r2_score(y_test, predicciones)

print("\n--- Resultados de la Evaluación ---")

print(f"Coeficiente de Determinación (R²): {r_cuadrado:.4f}")


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

mse = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse) 
mae = mean_absolute_error(y_test, predicciones)
r2 = r2_score(y_test, predicciones)

print("--- Evaluación Completa del Modelo ---")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, predicciones, alpha=0.6, color='blue')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Costo Médico Real ($)')
plt.ylabel('Costo Predicho por el Modelo ($)')
plt.title('Valores Reales vs. Predichos')

plt.subplot(1, 2, 2)

residuos = y_test - predicciones 
sns.histplot(residuos, kde=True, color='purple')
plt.xlabel('Error en la Predicción ($)')
plt.ylabel('Frecuencia')
plt.title('Distribución de los Residuos')

plt.tight_layout()
plt.show()



