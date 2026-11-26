import pandas as pd
import json
from pathlib import Path

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

carpeta = Path("artifacts")
carpeta.mkdir(parents=True, exist_ok=True)

def dataset():
    df = pd.read_csv(URL)
    return df


# Perfil del dataset
def observaciones(df):
    perfil = {
        "dataset" : "pinguins",
        "filas" : int(df.shape[0]),
        "columnas" : int(df.shape[1]),
        "nombre_columnas" : list(df.columns),
        "tipos_datos" : {col: str(tipo) for col, tipo in df.dtypes.items()},
        "valores_faltantes" : {col: int(cantidad) for col, cantidad in df.isnull().sum().items()},
        "duplicados" : int(df.duplicated().sum()),
        "cardinalidad" : {col: int(cantidad) for col, cantidad in df.nunique().items()}
    }
    return perfil    

# Guardar
def guardar_perfil(nombre, contenido):
    ruta = carpeta / nombre
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, indent=4, ensure_ascii=False)

    print(f"Perfil guardado en: {ruta}")

# ejecución
def ejecutar():
    df = dataset()
    perfil = observaciones(df)
    guardar_perfil("perfil.json", perfil)
    print("Ejecución completada.")

if __name__ == "__main__":
    ejecutar()
