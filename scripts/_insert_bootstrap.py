"""Inserta las celdas de bootstrap (Colab) en el notebook YA ejecutado, sin re-ejecutar.
Conserva todas las salidas existentes. Idempotente: no duplica si ya está.
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "challenge03_analitica_multidimensional.ipynb"

BOOT_MD = (
    "### Bootstrap (Google Colab / entornos sin el repo clonado)\n\n"
    "Este notebook importa código desde el paquete local `src/` y lee los CSV de `data/`. "
    "En **Google Colab** (o cualquier máquina sin el repositorio) esos archivos no existen y "
    "`from src import ...` fallaría. La siguiente celda lo resuelve automáticamente: si no "
    "encuentra el proyecto, **clona el repositorio de GitHub**, entra en él e instala las "
    "dependencias que Colab no trae (p. ej. `pmdarima`). En ejecución local no hace nada."
)

BOOT_CODE = r'''# --- Bootstrap portátil: hace que el notebook corra igual en local y en Colab ---
import os, sys, subprocess
from pathlib import Path

REPO_URL = "https://github.com/santiagovelcasallas/Challenge03_techlogistics-GomezVelez.git"
REPO_DIR = "Challenge03_techlogistics-GomezVelez"

def _project_here(p) -> bool:
    p = Path(p)
    return (p / "src").is_dir() and (p / "data").is_dir()

# ¿El proyecto ya está disponible (ejecución local dentro del repo)?
_found = any(_project_here(p) for p in [Path.cwd(), *Path.cwd().parents])

if not _found:
    print("Proyecto no encontrado localmente -> clonando desde GitHub (modo Colab)...")
    if not Path(REPO_DIR).is_dir():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
    os.chdir(REPO_DIR)
    # Dependencias que Colab no incluye por defecto (best-effort: el modelo P3
    # tiene fallback si pmdarima no queda disponible).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pmdarima"], check=False)
    print("Repositorio listo. Directorio de trabajo:", Path.cwd())
else:
    print("Proyecto encontrado localmente. Directorio de trabajo:", Path.cwd())'''

nb = nbf.read(NB, as_version=4)

# Idempotencia: si ya existe el bootstrap, no hacer nada
if any("Bootstrap portátil" in c.source for c in nb.cells):
    print("El bootstrap ya está presente; nada que hacer.")
    raise SystemExit(0)

# Localizar la celda markdown "### Configuración del entorno ..." e insertar antes
idx = next(i for i, c in enumerate(nb.cells)
           if c.cell_type == "markdown" and "Configuración del entorno" in c.source)

boot_md = nbf.v4.new_markdown_cell(BOOT_MD)
boot_code = nbf.v4.new_code_cell(BOOT_CODE)  # sin ejecutar: execution_count=None, outputs=[]

nb.cells[idx:idx] = [boot_md, boot_code]
nbf.write(nb, NB)
print(f"Bootstrap insertado en el índice {idx}. Total de celdas: {len(nb.cells)}")
