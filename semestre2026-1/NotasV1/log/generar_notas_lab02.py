"""
generar_notas_lab02.py
----------------------
El lab 02 tiene DOS secciones independientes con numeración solapada:
  S1 — Series de tiempo EUR/USD  (puntos 1-9)
  S2 — Breast Cancer Wisconsin   (puntos 1-14)
  Total: 23 tareas distintas

Fuente : EstudiantesOrdenado/  (estructura canónica Lab01/, Lab02/, …)
Salida : NotasV1/lab02_matriz.csv  (se sobreescribe en cada ejecución)
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ESTUDIANTES_DIR = ROOT / "EstudiantesOrdenado"
NOTAS_V1_DIR    = ROOT / "NotasV1"
OUTPUT_CSV      = NOTAS_V1_DIR / "lab02_matriz.csv"

POINT_PATTERN = re.compile(r"^\s*(\d+)[\).]\s+")

# Keywords que identifican cada sección a nivel de notebook completo
S1_KEYWORDS = re.compile(
    r"eurusd|eur/usd|eur.usd|diffprice|diff.price|series.de.tiempo|fitter|grouper|15d|1\s*mes|spread|apertura|cierre",
    re.IGNORECASE,
)
S2_KEYWORDS = re.compile(
    r"breast|radiusmean|diagnosisnumeric|benigno|maligno|diagnosis|violin|violín|zscore|z.score|iqr|outlier",
    re.IGNORECASE,
)

S1_POINTS = set(range(1, 10))   # 1-9
S2_POINTS = set(range(1, 15))   # 1-14
TOTAL_POINTS = len(S1_POINTS) + len(S2_POINTS)  # 23


# ---------------------------------------------------------------------------
# Búsqueda de notebooks: Lab02/ en cualquier nivel bajo el directorio del estudiante
# ---------------------------------------------------------------------------
# Búsqueda de notebooks: Lab02/ en cualquier nivel bajo el directorio del estudiante
# ---------------------------------------------------------------------------
def collect_lab02_notebooks(student_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in student_dir.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
        and "Lab02" in path.parts
    )


# ---------------------------------------------------------------------------
# Extracción de puntos por sección desde celdas markdown
# ---------------------------------------------------------------------------
def extract_points_by_section(notebook_path: Path) -> tuple[set[int], set[int]]:
    """Devuelve (s1_detected, s2_detected).

    Detección de sección a nivel de notebook completo (no por celda):
    - Escanea TODO el contenido para determinar qué secciones cubre.
    - Luego extrae los números de puntos de celdas markdown.
    - Asigna esos puntos a la(s) sección(es) detectadas.
    """
    data = json.loads(notebook_path.read_text(encoding="utf-8"))

    # Texto completo del notebook (markdown + código) para detectar secciones
    all_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in data.get("cells", [])
    )
    has_s1 = bool(S1_KEYWORDS.search(all_text))
    has_s2 = bool(S2_KEYWORDS.search(all_text))

    # Extraer todos los puntos numerados de celdas markdown
    all_points: set[int] = set()
    in_code_block = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        for raw_line in cell.get("source", []):
            line = raw_line.rstrip("\n")
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            mm = POINT_PATTERN.match(line)
            if mm:
                all_points.add(int(mm.group(1)))

    s1 = {n for n in all_points if n in S1_POINTS} if has_s1 else set()
    s2 = {n for n in all_points if n in S2_POINTS} if has_s2 else set()
    return s1, s2


# ---------------------------------------------------------------------------
# Construcción de la matriz
# ---------------------------------------------------------------------------
def build_matrix() -> list[dict]:
    student_dirs = sorted(p for p in ESTUDIANTES_DIR.iterdir() if p.is_dir())
    rows: list[dict] = []

    for student_dir in student_dirs:
        notebooks = collect_lab02_notebooks(student_dir)

        s1_det: set[int] = set()
        s2_det: set[int] = set()
        for nb in notebooks:
            a, b = extract_points_by_section(nb)
            s1_det |= a
            s2_det |= b

        row: dict = {"estudiante": student_dir.name}

        for n in range(1, 10):
            row[f"S1_P{n}"] = 1 if n in s1_det else 0
        for n in range(1, 15):
            row[f"S2_P{n}"] = 1 if n in s2_det else 0

        hits = len(s1_det) + len(s2_det)
        nota = round(min(5.0, (hits / TOTAL_POINTS) * 5.0), 2)
        row["nota_0_5"] = nota
        row["aprobado"] = "si" if nota >= 3.0 else "no"

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rows = build_matrix()

    fields = (
        ["estudiante"]
        + [f"S1_P{n}" for n in range(1, 10)]
        + [f"S2_P{n}" for n in range(1, 15)]
        + ["nota_0_5", "aprobado"]
    )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aprobados = sum(1 for r in rows if r["aprobado"] == "si")
    promedio  = round(sum(r["nota_0_5"] for r in rows) / len(rows), 2) if rows else 0.0

    print(f"Estudiantes  : {len(rows)}")
    print(f"Aprobados    : {aprobados}/{len(rows)}")
    print(f"Promedio     : {promedio:.2f}")
    print(f"Archivo CSV  : {OUTPUT_CSV.name}")


if __name__ == "__main__":
    main()
