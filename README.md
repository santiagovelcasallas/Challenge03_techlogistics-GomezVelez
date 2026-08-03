# TechLogistics S.A. — Challenge 03: Analítica Multidimensional

Análisis multicapa (CRISP-DM) sobre dos redes de sensores de **TechLogistics S.A.**:
**Agroindustria/Clima** (`agro_*`) y **Energía/Economía** (`ener_*`). El proyecto combina
**geo-visualización**, **procesamiento de señales** (FFT, filtrado Butterworth),
**diagnóstico de estacionariedad** (ADF), **análisis de redes** (centralidades) y
**modelado** (Granger, ARIMAX) para responder tres preguntas de negocio y cuatro de validación.

> Proyecto académico — EAFIT, Maestría en Ciencia de los Datos.
>
> **Nota sobre la numeración:** el PDF del docente se titula *"Challenge 02"*, pero corresponde
> al **workshop de la Lecture 03**. Se unifica la etiqueta a **Challenge 03** (el "02" fue un
> error de numeración en el documento fuente).

---

## 📁 Estructura del repositorio

```
Challenge03_techlogistics-GomezVelez/
├── data/                 # CSV (clean + noise) — versionados (pequeños, reproducibilidad)
├── notebooks/
│   └── challenge03_analitica_multidimensional.ipynb   # análisis completo (ejecutado)
├── src/                  # funciones reutilizables
│   ├── io_utils.py       # carga + DatetimeIndex sintético + diccionario de variables
│   ├── stationarity.py   # ADF, rolling stats, drift vs random walk
│   ├── signal_utils.py   # FFT/PSD, Butterworth, SNR, RMSE
│   ├── graph_utils.py    # grafo dirigido, degree/betweenness, cuello de botella
│   └── viz_utils.py      # scatter_mapbox geo, dibujo de grafos, export PNG
├── figures/              # PNG exportados (evidencia del informe)
├── reports/
│   └── informe_tecnico.pdf   # informe ejecutivo (generado programáticamente)
├── scripts/
│   ├── build_notebook.py # genera el notebook con nbformat (celdas MD + código)
│   └── build_report.py   # genera el PDF con ReportLab
├── requirements.txt
├── .gitignore
└── README.md
```

### ¿Por qué se versiona `data/`?
Los CSV son pequeños (~450 KB c/u) y son **necesarios para reproducir** el análisis de
principio a fin. Por eso se incluyen en el repositorio. Lo que **sí** se ignora (`.gitignore`)
es el entorno virtual (`venv/`), cachés y checkpoints.

---

## 🔧 Reproducción

Requiere **Python 3.11** (probado en 3.11.4). Las dependencias están **pinneadas a
versiones exactas** en `requirements.txt`; el pipeline completo se ejecuta de cero sin
errores con ese conjunto. Para una réplica byte-a-byte de todo el árbol, usar
`requirements-lock.txt`.

```bash
# 1) Crear y activar el entorno virtual
py -m venv venv
venv\Scripts\activate            # Windows PowerShell/CMD
# source venv/bin/activate       # Linux/macOS

# 2) Instalar dependencias (versiones exactas probadas)
pip install -r requirements.txt
#   alternativa exacta de todo el árbol:  pip install -r requirements-lock.txt

# 3) (Opcional) Regenerar el notebook desde el generador
py scripts/build_notebook.py

# 4) Ejecutar el notebook de principio a fin
jupyter nbconvert --to notebook --execute --inplace ^
  notebooks/challenge03_analitica_multidimensional.ipynb

# 5) Generar el informe PDF
py scripts/build_report.py
```

> ✅ **Verificado**: se recreó el entorno desde cero (`venv` limpio → `requirements.txt`) y
> se reejecutó todo el pipeline (58 celdas, **0 errores**) + generación del PDF. El `venv/`
> NO se versiona (está en `.gitignore`); cada quien lo reconstruye con los comandos de arriba.

---

## 🧭 Supuestos metodológicos (declarados)

1. **La señal `*_clean` SÍ existe.** El brief asumía que solo se entregaron los `*_noise`,
   pero `agro_clean.csv` y `ener_clean.csv` están presentes. Se usan como **referencia real
   (ground truth)** para SNR y RMSE (corrección documentada al supuesto original). Aun así se
   construye la versión *denoised* (Butterworth) para demostrar el pipeline de filtrado.
2. **Sin timestamp.** Los CSV no traen columna temporal. Se crea un **`DatetimeIndex`
   sintético horario** (`freq='h'`, inicio `2023-01-01`) para todo el análisis de series.
3. **Reproducibilidad.** Semilla de NumPy fija (`seed=42`).
4. **Diccionario de variables.** Cada variable se interpreta según el diccionario provisto.

---

## 🗺️ Fases y tareas (CRISP-DM)

| Fase | Tarea | Contenido |
|------|-------|-----------|
| 1 | T1 | Geo-visualización `scatter_mapbox` (color=NDVI, tamaño=Humedad) + clustering espacial |
| 1 | T2 | ADF sobre energía + media/varianza móvil; Drift vs Random Walk (Ener_5) |
| 2 | T3 | FFT/PSD de Ener_4, banda del ruido inyectado y **SNR real** |
| 2 | T4 | Butterworth pasa-bajo sobre Agro_3 (RH) + **RMSE** vs referencia |
| 3 | T5 | Grafos dirigidos, degree + **betweenness**, **nodo cuello de botella** |
| 4 | P1 | Granger `Ener_10 → Ener_9` y efecto de un fallo en el nodo puente |
| 4 | P2 | GPS filtrado, bajo NDVI vs alta pendiente (proxy Agro_10), inversión hídrica |
| 4 | P3 | ARIMAX de Demanda con Temperatura + centralidad; comparación de **AIC** |

---

## 🔑 Hallazgos clave

- **Estacionariedad (T2).** Las series de calidad de potencia (`Ener_8-10`) son estacionarias;
  los factores macro (`Ener_5-7`) no. El **Costo del Gas (Ener_5)** se comporta como un
  **Random Walk con Drift** (tendencia embebida + varianza móvil creciente).
- **Señales (T3–T4).** El ruido inyectado en `Ener_4` es de **alta frecuencia/banda ancha**;
  el **SNR real** se calcula directamente (≈18 dB; `Ener_4` es la serie-señal de menor SNR,
  la más ruidosa del grupo principal, aunque por encima del objetivo nominal de 5–12 dB por
  su gran amplitud cíclica). El **Butterworth pasa-bajo reduce el RMSE de `Agro_3` ~60 %**,
  recuperando la señal subyacente y mejorando la capacidad predictiva.
- **Redes (T5).** Ambas redes son **bipartitas** (Source y Target disjuntos, *single-hop*), por
  lo que la **betweenness es 0 para todos los nodos** — un hallazgo topológico, no un error. El
  **cuello de botella** se define entonces por **throughput** (grado ponderado); en la red de
  **despacho** eléctrico ese nodo de máxima carga es el punto crítico de estabilidad.
- **Negocio (P1–P3).** Existe relación causal (Granger) entre factor de potencia y voltaje,
  con implicaciones de estabilidad ante fallos del nodo puente; el bajo NDVI se asocia a zonas
  de alta pendiente (priorización de inversión hídrica); y la **centralidad del nodo mejora el
  ARIMAX** de demanda (menor AIC).

> Los valores numéricos exactos se calculan en el notebook y en el informe PDF
> (`reports/informe_tecnico.pdf`), reproducibles con los scripts de `scripts/`.

---

## 📊 Stack técnico

pandas · numpy · scipy · statsmodels · pmdarima · networkx · plotly · matplotlib · seaborn ·
scikit-learn · kaleido · nbformat/nbconvert · reportlab

> **Nota sobre exportación de figuras.** `kaleido` (export estático de Plotly) se incluye en
> `requirements.txt`, pero en este entorno Windows su primer `write_image` **bloquea el
> kernel**. Como alternativa equivalente y fiable, las figuras PNG de evidencia se exportan
> con **matplotlib**; la versión interactiva `scatter_mapbox` de Plotly se conserva en el
> notebook para exploración.
