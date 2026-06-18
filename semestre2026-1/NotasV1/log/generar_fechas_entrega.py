"""
generar_fechas_entrega.py
--------------------------
Lee el git log de Estudiantes/ y genera una tabla CSV con la fecha del
último commit que toca cada (estudiante, laboratorio).

Salida : NotasV1/fechas_entrega.csv
"""

from __future__ import annotations

import csv
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[1]
OUTPUT_CSV = ROOT / "NotasV1" / "fechas_entrega.csv"

# Detecta número de lab en un path (Lab01, Lab_01, laboratorio_01, lab01, etc.)
LAB_RE = re.compile(
    r"(?:laboratorio[s]?|lab)[_\s-]*0*(\d+)",
    re.IGNORECASE,
)


def get_git_log() -> str:
    result = subprocess.run(
        ["git", "log", "--format=COMMIT %H %ad", "--date=short",
         "--name-only", "--", "Estudiantes/"],
        capture_output=True, text=True, cwd=ROOT,
    )
    return result.stdout


def parse_log(raw: str) -> dict[tuple[str, str], str]:
    """
    Returns {(estudiante, 'Lab01'): 'YYYY-MM-DD (latest commit)'}.
    Latest commit = first seen in reverse-chronological git log.
    """
    latest: dict[tuple[str, str], str] = {}
    current_date = ""

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("COMMIT "):
            # COMMIT <hash> <date>
            parts = line.split()
            current_date = parts[2] if len(parts) >= 3 else ""
            continue
        if not line or line.startswith("COMMIT"):
            continue

        # Line is a file path, e.g.
        # semestre2026-1/Estudiantes/Garcia_10/Lab01/...
        # Strip leading repo folder prefix if present
        if "Estudiantes/" not in line:
            continue

        after = line.split("Estudiantes/", 1)[1]  # StudentName/...
        parts = after.split("/")
        if not parts:
            continue

        estudiante = parts[0]
        # Search for lab number anywhere in the remaining path
        remaining = "/".join(parts[1:])
        m = LAB_RE.search(remaining)
        if not m:
            # Also try filename itself
            m = LAB_RE.search(parts[-1] if len(parts) > 1 else "")
        if not m:
            continue

        lab_num = int(m.group(1))
        lab_key = f"Lab{lab_num:02d}"
        key = (estudiante, lab_key)

        # git log is newest-first → first occurrence = latest commit
        if key not in latest and current_date:
            latest[key] = current_date

    return latest


def build_table(latest: dict[tuple[str, str], str]) -> None:
    # Collect all students and labs
    students = sorted({e for e, _ in latest})
    labs     = sorted({l for _, l in latest}, key=lambda x: int(x[3:]))

    rows = []
    for est in students:
        row = {"estudiante": est}
        for lab in labs:
            row[lab] = latest.get((est, lab), "")
        rows.append(row)

    fields = ["estudiante"] + labs
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Estudiantes : {len(students)}")
    print(f"Labs        : {', '.join(labs)}")
    print(f"Archivo CSV : {OUTPUT_CSV.name}")
    print()
    # Pretty-print
    header = f"{'Estudiante':<16}" + "".join(f"  {l}" for l in labs)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = f"{row['estudiante']:<16}"
        for lab in labs:
            v = row.get(lab, "")
            line += f"  {v if v else '---       '}"
        print(line)


def main() -> None:
    raw = get_git_log()
    if not raw.strip():
        print("No se encontró historial git para Estudiantes/")
        return
    latest = parse_log(raw)
    build_table(latest)


if __name__ == "__main__":
    main()
