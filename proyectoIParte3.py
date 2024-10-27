import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carga el archivo retail_sales.csv en un DataFrame de Pandas
dataset = pd.read_csv('./data/retail_sales_dataset.csv')

# Normalización de las ventas entre 0 y 1 usando solo pandas
dataset['Normalized_Sales'] = (dataset['Total Amount'] - dataset['Total Amount'].min()) / (dataset['Total Amount'].max() - dataset['Total Amount'].min())

# Clasificación de ventas en categorías
def classify_sales(amount):
    if amount > 0.7:
        return 'Alta'
    elif amount > 0.3:
        return 'Media'
    else:
        return 'Baja'

dataset['Sales_Category'] = dataset['Normalized_Sales'].apply(classify_sales)


# verificar de que la columna Date sea del tipo datetime
dataset['Date'] = pd.to_datetime(dataset['Date'])

# Extraer el mes y añadirlo como nueva columna
dataset['Month'] = dataset['Date'].dt.month

# Agrupación por Producto y Tienda
grouped_product_store = dataset.groupby(['Product Category', 'Customer ID']).agg(
    Total_Sales=('Total Amount', 'sum'),
    Average_Quantity=('Quantity', 'mean'),
    Total_Transactions=('Transaction ID', 'count'),
    Min_Transaction_Value=('Total Amount', 'min'),
    Max_Transaction_Value=('Total Amount', 'max'),
    Std_Transaction_Value=('Total Amount', 'std'),
    Var_Transaction_Value=('Total Amount', 'var')
).reset_index()

# Agrupación por Producto y Mes
grouped_product_month = dataset.groupby(['Product Category', 'Month']).agg(
    Total_Sales=('Total Amount', 'sum'),
    Average_Quantity=('Quantity', 'mean'),
    Total_Transactions=('Transaction ID', 'count'),
    Min_Transaction_Value=('Total Amount', 'min'),
    Max_Transaction_Value=('Total Amount', 'max'),
    Std_Transaction_Value=('Total Amount', 'std'),
    Var_Transaction_Value=('Total Amount', 'var')
).reset_index()

# Mostrar los resultados de la agrupación por Producto y Tienda
print("Agrupación por Producto y Tienda:")
print(grouped_product_store.head())

# Mostrar los resultados de la agrupación por Producto y Mes
print("\nAgrupación por Producto y Mes:")
print(grouped_product_month.head())

# Función personalizada para calcular la desviación respecto a la media
def calculate_deviation(row):
    group_mean = row['Total_Sales']  # Media del grupo (se calculará después)
    deviation = row['Total Amount'] - group_mean
    return deviation

# Aplicar la función de desviación a cada grupo
def apply_deviation(group):
    group['Deviation'] = group['Total Amount'] - group['Total Amount'].mean()
    return group

# Aplicar la función personalizada a cada grupo
grouped_product_store = grouped_product_store.groupby(['Product Category', 'Customer ID']).apply(apply_deviation)

# Mostrar los resultados con la desviación
print("Resultados con desviación respecto a la media:")
print(grouped_product_store.head())

#A partir de aca es la Parte 4 del Proyecto

# Cálculo de estadísticas descriptivas básicas
descriptive_stats = dataset.describe()
print("Estadísticas descriptivas básicas:")
print(descriptive_stats)

#Histogramas y Boxplots
# Configuración general de estilo para seaborn
sns.set(style="whitegrid")

# Histograma para visualizar la distribución de Total Amount
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Total Amount'], kde=True, bins=30)
plt.title('Distribución de Total Amount')
plt.xlabel('Total Amount')
plt.ylabel('Frecuencia')
plt.show()

# Histograma para visualizar la distribución de Quantity
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Quantity'], kde=True, bins=30, color="skyblue")
plt.title('Distribución de Quantity')
plt.xlabel('Quantity')
plt.ylabel('Frecuencia')
plt.show()

# Histograma para visualizar la distribución de Normalized Sales
plt.figure(figsize=(10, 6))
sns.histplot(dataset['Normalized_Sales'], kde=True, bins=30, color="green")
plt.title('Distribución de Normalized Sales')
plt.xlabel('Normalized Sales')
plt.ylabel('Frecuencia')
plt.show()

# Boxplot para Total Amount
plt.figure(figsize=(10, 6))
sns.boxplot(x=dataset['Total Amount'])
plt.title('Boxplot de Total Amount')
plt.xlabel('Total Amount')
plt.show()

# Boxplot para Quantity
plt.figure(figsize=(10, 6))
sns.boxplot(x=dataset['Quantity'], color="skyblue")
plt.title('Boxplot de Quantity')
plt.xlabel('Quantity')
plt.show()

# Boxplot para Normalized Sales
plt.figure(figsize=(10, 6))
sns.boxplot(x=dataset['Normalized_Sales'], color="green")
plt.title('Boxplot de Normalized Sales')
plt.xlabel('Normalized Sales')
plt.show()

#Graficos de Lineas
# Agrupar las ventas por año y mes
monthly_sales = dataset.groupby(['Year', 'Month'])['Total Amount'].sum().reset_index()

# Configuración de estilo para seaborn
sns.set(style="whitegrid")

# Gráfico de líneas para la tendencia mensual de ventas
plt.figure(figsize=(14, 8))
sns.lineplot(data=monthly_sales, x='Month', y='Total Amount', hue='Year', marker="o", palette="tab10")
plt.title('Tendencia Mensual de Ventas')
plt.xlabel('Mes')
plt.ylabel('Total de Ventas')
plt.legend(title='Año')
plt.xticks(range(1, 13))
plt.show()

# Agrupar las ventas por año
annual_sales = dataset.groupby('Year')['Total Amount'].sum().reset_index()

#Graficos de Lineas

# Gráfico de líneas para la tendencia anual de ventas
plt.figure(figsize=(10, 6))
sns.lineplot(data=annual_sales, x='Year', y='Total Amount', marker="o", color="b")
plt.title('Tendencia Anual de Ventas')
plt.xlabel('Año')
plt.ylabel('Total de Ventas')
plt.show()

# Configuración general de estilo para seaborn
sns.set(style="whitegrid")

# Gráfico de dispersión entre Quantity y Total Amount
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity', y='Total Amount', data=dataset, color='blue')
plt.title('Relación entre Quantity y Total Amount')
plt.xlabel('Quantity')
plt.ylabel('Total Amount')
plt.show()

# Gráfico de dispersión entre Normalized Sales y Quantity
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity', y='Normalized_Sales', data=dataset, color='green')
plt.title('Relación entre Quantity y Normalized Sales')
plt.xlabel('Quantity')
plt.ylabel('Normalized Sales')
plt.show()

#Combinacion de Histogrmas y Boxplots

# Crear una figura con subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]})

# Boxplot en la parte superior
sns.boxplot(x=dataset['Total Amount'], ax=axes[0], color='lightblue')
axes[0].set(title='Boxplot de Total Amount', xlabel='')

# Histograma en la parte inferior
sns.histplot(dataset['Total Amount'], kde=True, bins=30, ax=axes[1], color='blue')
axes[1].set(title='Histograma de Total Amount', xlabel='Total Amount', ylabel='Frecuencia')

plt.tight_layout()
plt.show()

