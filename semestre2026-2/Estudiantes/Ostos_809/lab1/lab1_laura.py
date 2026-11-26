import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# FASE1
penguins=pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv")

# ===========================================================
# Parte A
# 1) Filas y Columnas
print("Filas y Columnas:")
print("hay " + str(penguins.shape[0]) + " Filas y hay " + str(penguins.shape[1]) + " columnas")# filas y columnas

# 2) tipos de valores: numericas y categoricas
print("\nTipos de Valores:")
print("Las variables categoricas aparecen como object y las numericas con float64")
print(penguins.dtypes) 

# 3) valores faltantes
print("\nValores faltantes:")
val_nulos=penguins.isnull().sum()
print(val_nulos)

# 4) filas duplicadas
print("\nFilas duplicadas:")
d=penguins.duplicated().sum()
print(" Hay " + str(d) + " filas duplicadas")

# 5) cardinalidad
print("\nCardinalidad:")
c=penguins.nunique()
print(c)

# ===========================================================
# Parte B
# 6) media, mediana, desviacion estandar y rango intercuartilico.
variables= penguins.select_dtypes(include=["float64"]).columns
print("\nVariables numericas:Estadisticos descriptivos")

for var in variables:
    media= penguins[var].mean()
    mediana= penguins[var].median()
    des= penguins[var].std()
    q1= penguins[var].quantile(0.25)
    q3= penguins[var].quantile(0.75)
    iqr= q3 - q1

    print("\nVariable: " + var)
    print("Media: " + str(media))
    print("Mediana: " + str(mediana))
    print("Desviacion estandar: " + str(des))
    print("Rango intercuartilico: " + str(iqr))

# 7) conteos y porcenatjes variables categoricas
categoricas= penguins.select_dtypes(include=["object"]).columns
print("\nVariables categoricas: conteos y porcentajes")

for var in categoricas:
    conteo= penguins[var].value_counts()
    porcentaje= penguins[var].value_counts(normalize=True)*100

    tabla1= pd.DataFrame({"Conteo": conteo, "porcentaje": porcentaje})
    print(tabla1)

# 8) Tablas cruzadas
print("\nTablas cruzadas")
print("tabla cruzada entre species y sex")
tabla2= pd.crosstab(penguins["species"], penguins["sex"])
print(tabla2)

print("\ntabla cruzada entre species y island")
tabla3= pd.crosstab(penguins["species"], penguins["island"])
print(tabla3)

print("\ntabla cruzada entre sex y island")
tabla4= pd.crosstab(penguins["island"],penguins["sex"])
print(tabla4)

# 9) Matrices de corelacion pearson y spearman
print("\nMatrices de correlacion")
print("Matriz de correlacion de Pearson")
matriz1= penguins[variables].corr(method="pearson")
print(matriz1)

print("\nMatriz de correlacion de Spearman")
matriz2= penguins[variables].corr(method="spearman")
print(matriz2)

# ===========================================================
# Parte C
# 10) Graficos de conteo de variables categoricas
print("\nGraficos de conteo de variables categoricas")
for var in categoricas: 
    plt.figure(figsize=(7,5))
    sns.countplot(data=penguins, x=var)
    plt.title("Grafico de conteo de " + var)
    plt.xlabel(var)
    plt.ylabel("Conteo")
    plt.show()

#. 11) Histogramas de variables numericas
print("\nHistogramas de varaibles numericas")
for var in variables:
    plt.figure(figsize=(7,5))
    sns.histplot(data=penguins, x=var, kde=True)
    plt.title(f"Histograma de {var}")
    plt.xlabel(var)
    plt.ylabel("Frecuencia")

    plt.savefig("histograma_" + var + ".png", dpi=300, bbox_inches="tight")

    plt.show()

# Dostrivució multimodal en las variables.

# 12) Boxplots de variables numericas
print("\nBoxplots: longitud del pico según especie")
plt.figure(figsize=(7,5))
sns.boxplot(data=penguins, x="species", y="bill_length_mm")
plt.title("Boxplot de longitud del pico según especie")
plt.xlabel("Especie")
plt.ylabel("Longitud del pico (mm)")

plt.savefig("boxplot_longitud_pico.png",dpi=300, bbox_inches="tight")
plt.show()

# 13) Scatter entre variables numericas
print("\nScatter entre variables numericas")
plt.figure(figsize=(7,5))
sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue="species")
plt.title("Scatter entre longitud y profundidad del pico según especie")
plt.xlabel("Longitud del pico (mm)")
plt.ylabel("Profundidad del pico (mm)")

plt.savefig("scatter_longitud_profundidad_pico.png",dpi=300, bbox_inches="tight")
plt.show()

# 14) Heatmap de correlacion
print("\nHeatmap de correlacion")
plt.figure(figsize=(7,5))
sns.heatmap(matriz1, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap de correlacion de Pearson")

plt.savefig("heatmap_correlacion_pearson.png", dpi=300, bbox_inches="tight")
plt.show()

# ===========================================================
# Parte D : Pruebas Estadisticas
# 15) Hipotesis Falsables
print("\nHipotesis Falsables")
print("\n1. H0: La longitud de la aleta y la masa corporal de los penguinos son independientes.\n   H1: Existe una asociacion entre la longitud de la aleta y la masa corporal.")
print("\n2. H0: No existe diferencias significativas en la longitud del pico entre las especies.\n   H1: Existe diferencias significativas en la longitud del pico entre las especies.")
print("\n3. H0: La especie y la isla son independientes.\n   H1: Existe una asociacion entre la especie y la isla.")

# Parte E: Pruebas Estadisticas
# 16) Seleccion de las pruebas estadisticas y 17) Reporte estadístico y p-valor
print("\nPruebas Estadisticas")

# Hipotesis 1: Pearson
datos_h1= penguins[["flipper_length_mm", "body_mass_g"]].dropna()
corr_pearson, p_pearson= stats.pearsonr(datos_h1["flipper_length_mm"], datos_h1["body_mass_g"])
print("\nHipotesis 1: Pearson")
print("Coeficiente de correlacion de Pearson: " , corr_pearson)
print("Valor p: " , p_pearson)
# No se rechaza H0 si p > 0.05.

# Hipotesis 1: spearman
rho_spearman, p_spearman= stats.spearmanr(datos_h1["flipper_length_mm"], datos_h1["body_mass_g"])
print("\nHipotesis 1: Spearman")
print("Coeficiente de correlacion de Spearman: ", rho_spearman)
print("Valor p: " , p_spearman)

# Hipotesis 2: ANOVA
adelie= penguins[penguins["species"]=="Adelie"]["bill_length_mm"].dropna()
chinstrap= penguins[penguins["species"]=="Chinstrap"]["bill_length_mm"].dropna()
gentoo= penguins[penguins["species"]=="Gentoo"]["bill_length_mm"].dropna()
f_anova, p_anova= stats.f_oneway(adelie, chinstrap, gentoo)
print("\nHipotesis 2: ANOVA")
print("Estadistico F: ", f_anova)
print("Valor p: ", p_anova)


# Hipotesis 3: Chi-cuadrado
tabla_chi = pd.crosstab(penguins["species"], penguins["island"])
chi2, p_chi, grados_libertad, esperados = stats.chi2_contingency(tabla_chi)
print("\nHipotesis 3: Chi-cuadrado")
print("Estadistico Chi-cuadrado: " , chi2)
print("Grados de libertad: " , grados_libertad)
print("p-valor: " , p_chi)
print("\n Frecuencias esperadas: \n", esperados)

# 18) verificar si la evidencia apoya o no la hipotesis. 
# hipotesis 1:
if p_pearson < 0.05:
    print("\nHipotesis 1 (Pearson): Se rechaza H0, existe una asociacion entre la longitud de la aleta y la masa corporal.")
else:
    print("\nHipotesis 1 (Pearson): No se rechaza H0, no existe una asociacion entre la longitud de la aleta y la masa corporal.")

# hipotesis 2:
if p_anova < 0.05:
    print("\nHipotesis 2 (ANOVA): Se rechaza H0, existe diferencias significativas en la longitud del pico entre las especies.")
else:
    print("\nHipotesis 2 (ANOVA): No se rechaza H0, no existe diferencias significativas en la longitud del pico entre las especies.")

# hipotesis 3:
if p_chi < 0.05:
    print("\nHipotesis 3 (Chi-cuadrado): Se rechaza H0, existe una asociacion entre la especie y la isla.")
else:
    print("\nHipotesis 3 (Chi-cuadrado): No se rechaza H0, no existe una asociacion entre la especie y la isla.")

# Organizar en tabla los resultados:
tabla= pd.DataFrame({
    "Hipótesis": [ "flipper_length_mm vs body_mass_g", "bill_length_mm según species", "species vs island"],
    "Prueba": ["Pearson", "ANOVA","Chi-cuadrado"],
    "Estadístico": [corr_pearson, f_anova, chi2],
    "p-valor": [p_pearson, p_anova, p_chi]
})

print("\nTabla de Pruebas:")
print(tabla)

# ============================================================

# Parte F: Conclusiones y Reporte Final

