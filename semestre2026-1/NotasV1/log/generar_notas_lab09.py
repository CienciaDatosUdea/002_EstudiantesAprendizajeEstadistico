"""
generar_notas_lab09.py
----------------------
Lab 09: Red neuronal desde cero — 4 puntos

Fuente : EstudiantesOrdenado/  (estructura canónica Lab09/)
Salida : NotasV1/lab09_matriz.csv  (se sobreescribe en cada ejecución)
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESTUDIANTES_DIR = ROOT / "EstudiantesOrdenado"
OUTPUT_CSV      = ROOT / "NotasV1" / "lab09_matriz.csv"

POINT_PATTERN = re.compile(r"^\s*(\d+)[\).]\s+")
LAB_FOLDER    = "Lab09"
NUM_PUNTOS    = 4


def collect_notebooks(student_dir: Path) -> list[Path]:
    return sorted(
        p for p in student_dir.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in p.parts
        and LAB_FOLDER in p.parts
    )


def extract_points(nb_path: Path) -> set[int]:
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    points: set[int] = set()
    in_code = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        for raw in cell.get("source", []):
            line = raw.rstrip("\n")
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            m = POINT_PATTERN.match(line)
            if m:
                points.add(int(m.group(1)))
    return points


def build_matrix() -> list[dict]:
    rows = []
    for student_dir in sorted(p for p in ESTUDIANTES_DIR.iterdir() if p.is_dir()):
        detected: set[int] = set()
        for nb in collect_notebooks(student_dir):
            detected.update(extract_points(nb))
        detected = {n for n in detected if 1 <= n <= NUM_PUNTOS}

        row = {"estudiante": student_dir.name}
        hits = 0
        for n in range(1, NUM_PUNTOS + 1):
            present = 1 if n in detected else 0
            row[f"P{n}"] = present
            hits += present

        nota = round(min(5.0, (hits / NUM_PUNTOS) * 5.0), 2)
        row["nota_0_5"] = nota
        row["aprobado"] = "si" if nota >= 3.0 else "no"
        rows.append(row)
    return rows


def main() -> None:
    rows = build_matrix()
    fields = ["estudiante"] + [f"P{n}" for n in range(1, NUM_PUNTOS + 1)] + ["nota_0_5", "aprobado"]
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
