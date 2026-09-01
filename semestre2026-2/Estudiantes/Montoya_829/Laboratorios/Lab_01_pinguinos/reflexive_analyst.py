"""Análisis reflexivo reproducible del conjunto de datos de pingüinos."""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import pearsonr
except ImportError:
    sys.exit(
        "Error: SciPy no está disponible. Instala SciPy en el entorno del curso para calcular Pearson."
    )


DATA_FILE = Path(__file__).resolve().parent / "data" / "penguins.csv"
LOG_FILE = Path(__file__).resolve().parent / "reflection_log.md"
MAX_CYCLES = 3


def cargar_observaciones_limpias(ruta):
    """Carga solo las columnas necesarias y omite filas con números faltantes."""
    observaciones = []
    with ruta.open(newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            try:
                largo = float(fila["bill_length_mm"])
                profundidad = float(fila["bill_depth_mm"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(largo) or not math.isfinite(profundidad):
                continue
            especie = (fila.get("species") or "").strip()
            if especie:
                observaciones.append(
                    {
                        "species": especie,
                        "bill_length_mm": largo,
                        "bill_depth_mm": profundidad,
                    }
                )
    if len(observaciones) < 3:
        raise ValueError("No hay suficientes observaciones numéricas válidas para calcular Pearson.")
    return observaciones


def calcular_pearson(observaciones):
    largos = [fila["bill_length_mm"] for fila in observaciones]
    profundidades = [fila["bill_depth_mm"] for fila in observaciones]
    if len(largos) < 2:
        raise ValueError("Se necesitan al menos dos observaciones para calcular Pearson.")
    resultado = pearsonr(largos, profundidades)
    return {"r": float(resultado.statistic), "p": float(resultado.pvalue), "n": len(largos)}


def construir_esquema_y_estadisticas(observaciones):
    """Devuelve un resumen sin exponer filas al generador."""
    resultado = calcular_pearson(observaciones)
    return (
        ["species", "bill_length_mm", "bill_depth_mm"],
        {
            "n": len(observaciones),
            "media_bill_length_mm": sum(fila["bill_length_mm"] for fila in observaciones)
            / len(observaciones),
            "media_bill_depth_mm": sum(fila["bill_depth_mm"] for fila in observaciones)
            / len(observaciones),
            "pearson": resultado,
        },
    )


def generar_hipotesis(esquema, estadisticas):
    """Rol generador: recibe solamente nombres de columnas y estadísticas agregadas."""
    if "bill_length_mm" not in esquema or "bill_depth_mm" not in esquema:
        raise ValueError("El esquema no contiene las variables requeridas.")
    resultado = estadisticas["pearson"]
    hipotesis = (
        "En la muestra completa, bill_length_mm y bill_depth_mm presentan una asociación lineal."
    )
    conclusion = (
        "En la muestra completa, la correlación de Pearson entre bill_length_mm y "
        f"bill_depth_mm fue r={resultado['r']:.3f}, p={resultado['p']:.3g} y n={resultado['n']}. "
        "Esto describe una asociación lineal del conjunto total y no demuestra causalidad."
    )
    return {"hipotesis": hipotesis, "conclusion": conclusion}


def criticar_conclusion(conclusion, descripcion_analisis):
    """Rol crítico: recibe texto, nunca observaciones ni filas del archivo."""
    descripcion = descripcion_analisis.lower()
    conclusion_normalizada = conclusion.lower()
    objeciones = []

    revisa_agrupacion = "por especie" in descripcion
    revisa_homogeneidad = "heterogene" in descripcion or revisa_agrupacion
    evita_causalidad = "no demuestra causalidad" in conclusion_normalizada or "no demuestran causalidad" in conclusion_normalizada

    if not revisa_agrupacion:
        objeciones.append(
            "Se ignoró la variable de agrupación species; la relación debe revisarse por especie."
        )
    if not revisa_homogeneidad:
        objeciones.append(
            "La muestra completa puede ser heterogénea y mezclar especies puede ocultar diferencias."
        )
    if not evita_causalidad:
        objeciones.append(
            "La conclusión afirma más de lo que Pearson permite, porque Pearson no demuestra causalidad."
        )
    return {"objeciones": objeciones, "bloqueante": bool(objeciones)}


def revisar_por_especie(observaciones, objeciones):
    """Rol revisor: recibe las observaciones limpias para responder a las objeciones."""
    if not objeciones:
        raise ValueError("No hay objeciones bloqueantes que revisar.")
    grupos = defaultdict(list)
    for fila in observaciones:
        grupos[fila["species"]].append(fila)

    resultados = {especie: calcular_pearson(filas) for especie, filas in sorted(grupos.items())}
    resultado_texto = "; ".join(
        f"{especie}: r={resultado['r']:.3f}, p={resultado['p']:.3g}, n={resultado['n']}"
        for especie, resultado in resultados.items()
    )
    conclusion = (
        f"Por especie, Pearson entre bill_length_mm y bill_depth_mm dio {resultado_texto}. "
        "Estos resultados describen asociaciones lineales dentro de cada especie y no demuestran causalidad."
    )
    return {
        "hipotesis": "La asociación entre las medidas puede cambiar entre especies.",
        "descripcion": (
            "Se usaron las filas numéricas válidas, se separaron por especie para reducir la "
            "heterogeneidad y se calculó Pearson dentro de cada grupo."
        ),
        "resultado": resultado_texto,
        "conclusion": conclusion,
    }


def escribir_log(ciclos):
    lineas = ["# Registro de reflexión", ""]
    for ciclo in ciclos:
        lineas.extend(
            [
                f"## Ciclo {ciclo['numero']}",
                f"**Hipótesis:** {ciclo['hipotesis']}",
                f"**Análisis ejecutado:** {ciclo['descripcion']}",
                f"**Resultado numérico:** {ciclo['resultado']}",
                "**Objeciones:**",
            ]
        )
        if ciclo["critica"]["objeciones"]:
            lineas.extend(f"- {objecion}" for objecion in ciclo["critica"]["objeciones"])
        else:
            lineas.append("- Ninguna.")
        lineas.extend(
            [
                f"**Bloqueante:** {str(ciclo['critica']['bloqueante']).lower()}",
                f"**Qué cambió:** {ciclo['cambio']}",
                f"**Conclusión:** {ciclo['conclusion']}",
                "",
            ]
        )
    LOG_FILE.write_text("\n".join(lineas), encoding="utf-8")


def main():
    observaciones = cargar_observaciones_limpias(DATA_FILE)
    esquema, estadisticas = construir_esquema_y_estadisticas(observaciones)
    generado = generar_hipotesis(esquema, estadisticas)
    descripcion_inicial = (
        "Se eliminaron filas con bill_length_mm o bill_depth_mm faltantes y se calculó Pearson "
        "sobre toda la muestra válida."
    )
    resultado_inicial = (
        f"r={estadisticas['pearson']['r']:.3f}, p={estadisticas['pearson']['p']:.3g}, "
        f"n={estadisticas['pearson']['n']}"
    )
    ciclos = [
        {
            "numero": 1,
            "hipotesis": generado["hipotesis"],
            "descripcion": descripcion_inicial,
            "resultado": resultado_inicial,
            "conclusion": generado["conclusion"],
            "critica": criticar_conclusion(generado["conclusion"], descripcion_inicial),
            "cambio": "Análisis inicial de la muestra completa.",
        }
    ]

    while ciclos[-1]["critica"]["bloqueante"] and len(ciclos) < MAX_CYCLES:
        revisado = revisar_por_especie(observaciones, ciclos[-1]["critica"]["objeciones"])
        ciclos.append(
            {
                "numero": len(ciclos) + 1,
                "hipotesis": revisado["hipotesis"],
                "descripcion": revisado["descripcion"],
                "resultado": revisado["resultado"],
                "conclusion": revisado["conclusion"],
                "critica": criticar_conclusion(revisado["conclusion"], revisado["descripcion"]),
                "cambio": "Se repitió Pearson por especie y se evitó una afirmación causal.",
            }
        )

    escribir_log(ciclos)
    if ciclos[-1]["critica"]["bloqueante"]:
        raise RuntimeError("La revisión no resolvió las objeciones en el máximo de tres ciclos.")
    print(f"Registro creado en {LOG_FILE} con {len(ciclos)} ciclos.")


if __name__ == "__main__":
    main()
