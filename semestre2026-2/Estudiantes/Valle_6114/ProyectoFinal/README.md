# Cuatro noches para contar mosquitos

Mateo Caballero Serna, Lucian Valle Aguilar

- `EstimacionFermi_00/`: punto 1 (documento, notebook y figura).
- `PlanAnalisis_01/`: plan de análisis, síntesis, hoja de campo y campañas reservadas.
- `SintesisFinal_02/`: notebook final con los puntos 2, 3 y 4, datos y figuras.

## Cómo correrlo

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

En VS Code (con la extensión Jupyter) abran `SintesisFinal_02/notebook_final.ipynb`, elijan el kernel `.venv` y corran todo. Lee los datos de `data/` y guarda las figuras en `figuras/`.

Datos: Ahumada ML et al. (2023), INS, doi:10.15472/kftrcn, licencia CC BY-NC 4.0.

## Pendiente
- Presentación.
- Reflexión individual de cada uno.
- Exportar `estimacion_fermi.md` y `plan_analisis_sintesis.md` a PDF y revisar que no pasen de 2 páginas.
