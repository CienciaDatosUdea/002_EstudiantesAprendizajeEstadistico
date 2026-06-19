"""
generar_notas_global.py
-----------------------
Genera la tabla acumulada de notas de todos los laboratorios:

  filas    → estudiantes
  columnas → Lab01, Lab02, …  + promedio_global

Lee cada lab0X_matriz.csv generado por los scripts individuales.
Si no existe el CSV del lab, la nota queda como 0.0.

Fuente : EstudiantesOrdenado/
Salida : NotasV1/notas_global.csv  (se sobreescribe en cada ejecución)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
ESTUDIANTES_DIR = ROOT / "EstudiantesOrdenado"
NOTAS_V1_DIR    = ROOT / "NotasV1"
OUTPUT_CSV      = NOTAS_V1_DIR / "notas_global.csv"


# ---------------------------------------------------------------------------
# Descubrir laboratorios disponibles (por archivos de matriz sin versión)
# ---------------------------------------------------------------------------
def discover_lab_matrices() -> dict[int, Path]:
    """Devuelve {lab_num: path} para cada lab0X_matriz.csv presente."""
    labs: dict[int, Path] = {}
    for path in NOTAS_V1_DIR.glob("lab*_matriz.csv"):
        m = re.match(r"lab0*(\d+)_matriz\.csv", path.name, re.IGNORECASE)
        if m:
            labs[int(m.group(1))] = path
    return dict(sorted(labs.items()))


# ---------------------------------------------------------------------------
# Leer nota por estudiante desde un CSV de matriz
# ---------------------------------------------------------------------------
def read_notas_from_matrix(csv_path: Path) -> dict[str, float]:
    """Devuelve {estudiante: nota_0_5}."""
    result: dict[str, float] = {}
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            result[row["estudiante"]] = float(row.get("nota_0_5", 0.0))
    return result


# ---------------------------------------------------------------------------
# Construcción de la tabla global
# ---------------------------------------------------------------------------
def build_global() -> tuple[list[dict], list[int]]:
    lab_matrices = discover_lab_matrices()
    lab_numbers  = list(lab_matrices.keys())

    # All notas per lab
    notas_per_lab: dict[int, dict[str, float]] = {
        lab_num: read_notas_from_matrix(path)
        for lab_num, path in lab_matrices.items()
    }

    student_dirs = sorted(p for p in ESTUDIANTES_DIR.iterdir() if p.is_dir())
    rows: list[dict] = []

    for student_dir in student_dirs:
        student = student_dir.name
        row: dict = {"estudiante": student}
        notas = []

        for lab_num in lab_numbers:
            nota = notas_per_lab[lab_num].get(student, 0.0)
            row[f"Lab{lab_num:02d}"] = nota
            notas.append(nota)

        row["promedio"] = round(sum(notas) / len(notas), 2) if notas else 0.0
        rows.append(row)

    return rows, lab_numbers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rows, lab_numbers = build_global()

    lab_cols = [f"Lab{n:02d}" for n in lab_numbers]
    fields   = ["estudiante"] + lab_cols + ["promedio"]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Labs incluidos : {', '.join(lab_cols)}")
    print(f"Estudiantes    : {len(rows)}")
    if rows:
        promedio_global = round(sum(r["promedio"] for r in rows) / len(rows), 2)
        print(f"Promedio global: {promedio_global:.2f}")
    print(f"Archivo CSV    : {OUTPUT_CSV.name}")


if __name__ == "__main__":
    main()
