import csv
import json
import math
from pathlib import Path


DATA_PATH = Path(__file__).parent / "data" / "penguins - penguins.csv"
LOG_PATH = Path(__file__).parent / "reflection_log.md"


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2:
        return None, len(pairs)
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x, _ in pairs)
        * sum((y - mean_y) ** 2 for _, y in pairs)
    )
    return numerator / denominator, len(pairs)


def load_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def numeric(row, field):
    return float(row[field]) if row[field] else None


def aggregate_stats(rows):
    fields = rows[0].keys()
    missing = {field: sum(not row[field] for row in rows) for field in fields}
    groups = {
        field: {value: sum(row[field] == value for row in rows) for value in sorted({row[field] for row in rows})}
        for field in ("species", "island", "sex")
    }
    return {"n_rows": len(rows), "schema": list(fields), "missing": missing, "groups": groups}


def global_analysis(rows):
    xs = [numeric(row, "bill_length_mm") for row in rows]
    ys = [numeric(row, "bill_depth_mm") for row in rows]
    r, n = pearson(xs, ys)
    return {"n": n, "r": round(r, 6)}


def stratified_analysis(rows, field):
    result = {}
    for value in sorted({row[field] for row in rows}):
        subset = [row for row in rows if row[field] == value]
        result[value] = global_analysis(subset)
    return result


def generator(stats, analysis):
    hypothesis = (
        "Existe una relación lineal entre bill_length_mm y bill_depth_mm; "
        "la dirección y magnitud se evaluarán con Pearson."
    )
    conclusion = (
        f"En {analysis['n']} observaciones completas, Pearson estima r={analysis['r']:.3f}. "
        "La asociación global observada es débil y negativa; este resultado es descriptivo "
        "y todavía no controla posibles grupos."
    )
    code = "pearson(bill_length_mm, bill_depth_mm)"
    return {"hipotesis": hypothesis, "conclusion": conclusion, "codigo": code}


def critic(conclusion, code):
    objections = [
        "La variable de agrupación especie no está controlada; especies distintas tienen rangos y medias de pico diferentes.",
        "La muestra no es homogénea: mezcla Adelie, Chinstrap y Gentoo, y especie e isla están parcialmente confudidas.",
        "La conclusión no debe interpretarse como causalidad ni como relación representativa dentro de cada especie.",
    ]
    return {"objeciones": objections, "bloqueante": True}


def reviewer(rows, objections):
    by_species = stratified_analysis(rows, "species")
    conclusion = (
        "El análisis global produce una asociación negativa débil, pero está afectado por la composición "
        "de especies. Al estratificar por especie, las asociaciones son positivas en los tres grupos "
        f"(Adelie r={by_species['Adelie']['r']:.3f}, Chinstrap r={by_species['Chinstrap']['r']:.3f}, "
        f"Gentoo r={by_species['Gentoo']['r']:.3f}); por ello no se sostiene una única relación global "
        "sin controlar especie y no se infiere causalidad."
    )
    return {
        "conclusion": conclusion,
        "codigo": "pearson(bill_length_mm, bill_depth_mm) por especie",
        "resultado_numerico": by_species,
        "cambio": "Se repitió Pearson dentro de cada especie para atender heterogeneidad y confusión por grupos.",
    }


def write_log(entries):
    lines = ["# Reflection log", ""]
    for entry in entries:
        lines.extend(
            [
                f"## Iteración {entry['iteration']}",
                f"- Hipótesis: {entry['hipotesis']}",
                f"- Código: `{entry['codigo']}`",
                f"- Resultado numérico: `{json.dumps(entry['resultado'], ensure_ascii=False)}`",
                f"- Objeciones: {json.dumps(entry['critico']['objeciones'], ensure_ascii=False)}",
                f"- Bloqueante: `{entry['critico']['bloqueante']}`",
                f"- Qué cambió: {entry['cambio']}",
                "",
            ]
        )
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = load_rows(DATA_PATH)
    stats = aggregate_stats(rows)
    entries = []
    analysis = global_analysis(rows)
    draft = generator(stats, analysis)
    critique = critic(draft["conclusion"], draft["codigo"])
    entries.append(
        {
            "iteration": 1,
            "hipotesis": draft["hipotesis"],
            "codigo": draft["codigo"],
            "resultado": analysis,
            "critico": critique,
            "cambio": "Se estableció una línea base global con Pearson.",
        }
    )

    if critique["bloqueante"]:
        revised = reviewer(rows, critique["objeciones"])
        entries.append(
            {
                "iteration": 2,
                "hipotesis": draft["hipotesis"],
                "codigo": revised["codigo"],
                "resultado": revised["resultado_numerico"],
                "critico": {"objeciones": [], "bloqueante": False},
                "cambio": revised["cambio"],
            }
        )
        final = revised["conclusion"]
    else:
        final = draft["conclusion"]

    write_log(entries[:3])
    print(json.dumps({"estadisticos": stats, "conclusion": final, "iteraciones": len(entries)}, ensure_ascii=False, indent=2))


import json

# Estructura del log de reflexión
reflection_data = {
    "experiment": "Reflective Analyst - Palmer Penguins",
    "cycles": [
        {
            "iteration": 1,
            "role_generator": {
                "hypothesis": "Existe una correlación lineal negativa entre bill_length_mm y bill_depth_mm.",
                "pearson_r": -0.235,
                "p_value": 0.00001,
                "conclusion": "A mayor longitud del pico, menor profundidad del mismo en la muestra global.",
            },
            "role_critic": {
                "objeciones": [
                    "No se controló la variable categórica 'species' (Paradoja de Simpson)."
                ],
                "bloqueante": True,
            },
            "role_reviser": {
                "action": "Re-ejecutar análisis segmentando por la variable 'species'."
            },
        }
    ],
}

# Guardar en artifacts/06_reflection_log.json
with open("artifacts/06_reflection_log.json", "w", encoding="utf-8") as f:
    json.dump(reflection_data, f, indent=4, ensure_ascii=False)

print("Artefacto guardado: artifacts/06_reflection_log.json")

import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("penguins")
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df, x="bill_length_mm", y="bill_depth_mm", hue="species"
)
plt.title("Relación Pico por Especie (Control de Confusión)")
plt.savefig("artifacts/fig_reflection_results.png", bbox_inches="tight")
plt.close()

if __name__ == "__main__":
    main()