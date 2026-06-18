"""
ordenar_estudiantes.py
----------------------
1. Copia Estudiantes/ → EstudiantesOrdenado/  (solo si no existe)
2. Dentro de cada directorio de estudiante en EstudiantesOrdenado/,
   renombra todos los subdirectorios que correspondan a un laboratorio
   a la forma canónica: Lab01, Lab02, Lab03, …

Reglas:
- Solo renombra directorios (nunca archivos).
- No borra nada.
- Si el directorio destino ya existe, reporta conflicto y lo omite.
- No recursa dentro de un directorio que ya es un lab (evita renombrar
  subdirectorios internos como Entregables, artifacts, etc.).
- Reconoce: Laboratorio_1, Lab_02, LAB3_PANDAS, sol_Lab_04, lab 5,
  Lab_01_pinguinos, Laboratorio_01, Laboratorios/Lab_2, etc.

Patrones reconocidos:
  [sol_] (laboratorio[s] | lab) [separador] [0*] NUMERO
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[1]
ORIGEN     = ROOT / "Estudiantes"
DESTINO    = ROOT / "EstudiantesOrdenado"

# Patrón para reconocer un directorio como lab N
LAB_PATTERN = re.compile(
    r"^(?:sol[_\s-]+)?(?:laboratorio[s]?|lab)[_\s-]*0*(\d+)(?!\d)",
    re.IGNORECASE,
)


def get_lab_num(name: str) -> int | None:
    """Devuelve el número de laboratorio si el nombre coincide, o None."""
    m = LAB_PATTERN.match(name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Renombrado recursivo (top-down)
# ---------------------------------------------------------------------------
def rename_lab_dirs(base: Path, stats: dict) -> None:
    """
    Recorre `base` buscando subdirectorios con nombre de lab.
    - Si encuentra uno: lo renombra a Lab{N:02d} y NO recursa dentro
      (sus subdirectorios internos conservan sus nombres).
    - Si NO es lab: recursa dentro para seguir buscando.
    """
    try:
        children = sorted(base.iterdir())
    except PermissionError:
        return

    for child in children:
        if not child.is_dir():
            continue
        if child.name.startswith(".") or ".ipynb_checkpoints" in child.parts:
            continue

        lab_num = get_lab_num(child.name)

        if lab_num is not None:
            new_name = f"Lab{lab_num:02d}"
            if child.name == new_name:
                stats["ya_correcto"] += 1
                # También NO recursamos dentro para no tocar internos
                continue

            new_path = child.parent / new_name
            if new_path.exists():
                print(f"  CONFLICTO  {child.relative_to(DESTINO)}  →  {new_name}  (ya existe)")
                stats["conflicto"] += 1
                continue

            child.rename(new_path)
            print(f"  RENOMBRADO {child.relative_to(DESTINO)}  →  {new_name}")
            stats["renombrado"] += 1
            # No recursamos dentro del directorio de lab
        else:
            # Directorio contenedor (ej. Laboratorios, LABORATORIOS, laboratorios)
            # → entramos a buscar labs adentro
            rename_lab_dirs(child, stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # 1. Copiar si no existe
    if not DESTINO.exists():
        print(f"Copiando {ORIGEN.name} → {DESTINO.name} …")
        shutil.copytree(ORIGEN, DESTINO)
        print("Copia completa.\n")
    else:
        print(f"EstudiantesOrdenado ya existe — trabajando sobre la copia existente.\n")

    stats = {"renombrado": 0, "ya_correcto": 0, "conflicto": 0}

    student_dirs = sorted(p for p in DESTINO.iterdir() if p.is_dir() and not p.name.startswith("."))
    for student_dir in student_dirs:
        print(f"[{student_dir.name}]")
        rename_lab_dirs(student_dir, stats)

    print(f"\n{'='*55}")
    print(f"  Renombrados    : {stats['renombrado']}")
    print(f"  Ya correctos   : {stats['ya_correcto']}")
    print(f"  Conflictos     : {stats['conflicto']}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
