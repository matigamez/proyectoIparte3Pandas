import pandas as pd

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