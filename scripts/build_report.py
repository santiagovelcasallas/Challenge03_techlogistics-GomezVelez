"""Generate the executive technical report (PDF) programmatically with ReportLab.

Recomputes the key metrics from the reusable ``src`` modules (so the PDF is
self-contained and reproducible) and embeds the figures exported to ``figures/``.
Answers the three business questions (P1-P3) and the four validation questions.

Usage:
    py scripts/build_report.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src import io_utils, stationarity, signal_utils, graph_utils  # noqa: E402

from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # noqa: E402
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,  # noqa: E402
                                Table, TableStyle, PageBreak)

DATA = ROOT / "data"
FIGS = ROOT / "figures"
OUT = ROOT / "reports" / "informe_tecnico.pdf"

# --------------------------------------------------------------- compute metrics
io_utils.set_seeds(42)
agro_clean, agro_noise = io_utils.load_pair(DATA, "agro")
ener_clean, ener_noise = io_utils.load_pair(DATA, "ener")

# T2 drift/random walk
diag_gas = stationarity.classify_drift_vs_randomwalk(ener_noise["Ener_5"])
adf_ener = stationarity.adf_table(ener_noise, io_utils.value_columns(ener_noise, "Ener"),
                                  io_utils.ENER_NAMES)

# T3 SNR
snr_ener4 = signal_utils.snr_db(ener_clean["Ener_4"], ener_noise["Ener_4"])

# T4 Butterworth RMSE
agro3_filt = signal_utils.butter_lowpass(agro_noise["Agro_3"], cutoff=0.05, fs=1.0, order=4)
rmse_noise = signal_utils.rmse(agro_noise["Agro_3"], agro_clean["Agro_3"])
rmse_filt = signal_utils.rmse(agro3_filt, agro_clean["Agro_3"])
rmse_improve = (1 - rmse_filt / rmse_noise) * 100

# T5 graphs
G_agro = graph_utils.build_directed_graph(agro_noise)
G_ener = graph_utils.build_directed_graph(ener_noise)
bn_agro, bv_agro, method_agro = graph_utils.bottleneck_node(G_agro)
bn_ener, bv_ener, method_ener = graph_utils.bottleneck_node(G_ener)
btw_degenerate = graph_utils.is_betweenness_degenerate(G_ener)

# P1 Granger
from statsmodels.tsa.stattools import grangercausalitytests
gc = grangercausalitytests(ener_noise[["Ener_9", "Ener_10"]].dropna(), maxlag=5, verbose=False)
gc_p = {lag: gc[lag][0]["ssr_ftest"][1] for lag in range(1, 6)}
gc_best_lag = min(gc_p, key=gc_p.get)
gc_best_p = gc_p[gc_best_lag]

# P2 Spearman NDVI vs slope proxy
from scipy.stats import spearmanr
agro = agro_noise.copy()
agro["lat_grid"] = agro["Latitude"].round(2)
agro["lon_grid"] = agro["Longitude"].round(2)
grid = (agro.groupby(["lat_grid", "lon_grid"])
             .agg(ndvi=("Agro_5", "mean"), slope=("Agro_10", "mean")).reset_index())
rho, rho_p = spearmanr(grid["ndvi"], grid["slope"])

# P3 ARIMAX AIC (fixed order for speed/reproducibility of the report)
from statsmodels.tsa.statespace.sarimax import SARIMAX
deg_cent = graph_utils.node_centrality_map(G_ener, kind="degree")
ener = ener_noise.copy()
ener["src_centrality"] = ener["Source_Node"].astype(int).map(deg_cent).fillna(0.0)
y = ener["Ener_1"].astype(float)
order = (2, 0, 2)
aic_base = SARIMAX(y, exog=ener[["Ener_3"]], order=order,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).aic
m_full = SARIMAX(y, exog=ener[["Ener_3", "src_centrality"]], order=order,
                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
aic_full = m_full.aic
delta_aic = aic_base - aic_full

# --------------------------------------------------------------- PDF styling
styles = getSampleStyleSheet()
styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], textColor=colors.HexColor("#1a3c6e"),
                          spaceBefore=14, spaceAfter=8))
styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], textColor=colors.HexColor("#255a9e"),
                          spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13,
                          alignment=4))  # justified
styles.add(ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, leading=10,
                          textColor=colors.grey))
styles.add(ParagraphStyle("KPI", parent=styles["BodyText"], fontSize=9.5, leading=13))

story = []
P = lambda t, s="Body": story.append(Paragraph(t, styles[s]))  # noqa: E731
SP = lambda h=0.3: story.append(Spacer(1, h * cm))  # noqa: E731


def add_figure(name: str, caption: str, width=15.5):
    path = FIGS / name
    if path.exists():
        img = Image(str(path))
        ratio = img.imageHeight / img.imageWidth
        img.drawWidth = width * cm
        img.drawHeight = width * ratio * cm
        story.append(img)
        story.append(Paragraph(caption, styles["Small"]))
        SP(0.4)
    else:
        P(f"[Figura {name} no encontrada — ejecutar el notebook primero]", "Small")


# --------------------------------------------------------------- cover
P("Informe Técnico Ejecutivo", "Title")
P("Challenge 02 — Analítica Multidimensional", "H2c")
P("TechLogistics S.A. · Metodología CRISP-DM · EAFIT — Maestría en Ciencia de los Datos", "Small")
SP(0.3)
P("Este informe sintetiza los hallazgos del análisis multicapa sobre las redes de "
  "sensores de <b>Agroindustria/Clima</b> y <b>Energía/Economía</b>. Combina procesamiento "
  "de señales (FFT, filtrado Butterworth), diagnóstico de estacionariedad (ADF), análisis "
  "de redes (centralidades) y modelado (Granger, ARIMAX) para responder tres preguntas de "
  "negocio y cuatro de validación.")
SP(0.2)

# Assumptions box
assum = ("<b>Supuestos metodológicos.</b> (1) Los archivos <i>*_clean</i> SÍ existen, por lo "
         "que se usan como <b>referencia real</b> para SNR/RMSE (corrección al supuesto del "
         "brief). (2) Sin columna temporal: se construye un <b>DatetimeIndex sintético "
         "horario</b> (inicio 2023-01-01). (3) Semilla NumPy fija (42). (4) Interpretación "
         "según el diccionario de variables provisto.")
story.append(Table([[Paragraph(assum, styles["Small"])]], colWidths=[16 * cm],
                    style=TableStyle([
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#255a9e")),
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3fb")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6)])))
SP(0.4)

# KPI table
kpi_rows = [
    ["Métrica", "Valor"],
    ["SNR real de Ener_4 (Gen. Eólica)", f"{snr_ener4:.2f} dB"],
    ["RMSE Agro_3: noise -> filtrada (Butterworth)", f"{rmse_noise:.3f} -> {rmse_filt:.3f} ({rmse_improve:.1f}% menor)"],
    ["Costo del Gas (Ener_5)", diag_gas["verdict"]],
    ["Nodo cuello de botella — ENERGÍA", f"nodo {bn_ener} ({method_ener}={bv_ener:.0f} reg.)"],
    ["Nodo cuello de botella — AGRO", f"nodo {bn_agro} ({method_agro}={bv_agro:.0f} reg.)"],
    ["Granger Ener_10 -> Ener_9 (mejor lag)", f"lag {gc_best_lag}, p={gc_best_p:.4f}"],
    ["Spearman(NDVI, pendiente-proxy)", f"rho={rho:.3f} (p={rho_p:.3g})"],
    ["ARIMAX AIC sin / con centralidad", f"{aic_base:.1f} / {aic_full:.1f} (Δ={delta_aic:.1f})"],
]
t = Table(kpi_rows, colWidths=[9.5 * cm, 6.5 * cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c6e")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d6ea")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fc")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
P("Tablero de indicadores clave", "H2c")
story.append(t)
story.append(PageBreak())

# --------------------------------------------------------------- P1
P("Preguntas de Negocio", "H1c")
P("P1 · Causalidad de Granger y estabilidad de la red", "H2c")
concl_g = ("SÍ existe causalidad de Granger" if gc_best_p < 0.05
           else "NO se detecta causalidad de Granger significativa")
P(f"Se evaluó si el <b>Factor de Potencia (Ener_10)</b> causa-Granger al <b>Voltaje "
  f"(Ener_9)</b> (ambas estacionarias, calidad de potencia). Resultado: <b>{concl_g}</b> "
  f"(mejor lag {gc_best_lag}, p={gc_best_p:.4f}). "
  "Interpretación: cuando existe, el pasado del factor de potencia aporta información "
  "predictiva sobre el voltaje. Cruzando con el análisis de red (T5), un <b>fallo en el nodo "
  f"de mayor throughput (nodo {bn_ener} en la red de despacho)</b> que degrade el factor de "
  "potencia se propagaría —vía esta relación causal— como <b>inestabilidad de voltaje</b> en "
  "todos los destinos que alimenta. Como ese nodo canaliza la mayor fracción de registros de "
  "despacho, la perturbación afecta a más carga antes de poder aislarse, elevando el riesgo "
  "sistémico. <b>Mitigación:</b> compensación reactiva (bancos de capacitores) y redundancia "
  "física en el nodo de mayor carga.")
P("<b>Nota topológica.</b> La red es <b>bipartita</b> (Source y Target son conjuntos "
  "disjuntos, single-hop), por lo que la <b>betweenness es 0 para todos los nodos</b> "
  "(ningún nodo es intermediario). El cuello de botella se define entonces por "
  "<b>throughput</b> (grado ponderado = registros que pasan por el nodo), criterio con "
  "sentido operativo directo.", "Small")
add_figure("t5_grafos_betweenness.png",
           "Figura P1/T5. Redes dirigidas bipartitas; en rojo el nodo cuello de botella (mayor throughput).")

# --------------------------------------------------------------- P2
P("P2 · Bajo NDVI, pendiente y recomendación de inversión hídrica", "H2c")
signo = "negativa (a menor NDVI, mayor pendiente)" if rho < 0 else "positiva"
P(f"Tras <b>filtrar el ruido de coordenadas GPS</b> (agregación en rejilla ~1 km), se "
  f"relacionó el NDVI medio por celda con la <b>varianza del viento (Agro_10) como proxy de "
  f"pendiente</b>. La correlación de Spearman es <b>rho={rho:.3f}</b> (p={rho_p:.3g}), una "
  f"relación {signo}. Los sensores de bajo NDVI tienden a ubicarse en celdas de mayor "
  "pendiente, donde la escorrentía y la baja retención hídrica limitan la biomasa. "
  "<b>Recomendación:</b> priorizar geográficamente la inversión en infraestructura hídrica "
  "(riego por goteo, terrazas y zanjas de infiltración, cosecha de agua) en las celdas de "
  "ladera identificadas, en lugar de una distribución uniforme: allí el retorno marginal "
  "sobre NDVI/biomasa es máximo.")
add_figure("p2_ndvi_vs_slope.png",
           "Figura P2. NDVI por celda GPS frente al proxy de pendiente (varianza del viento).", 11)
add_figure("t1_geo_ndvi.png",
           "Figura T1. Distribución geográfica de sensores (color=NDVI, tamaño=Humedad).")

# --------------------------------------------------------------- P3
story.append(PageBreak())
P("P3 · ARIMAX de la Demanda con centralidad del nodo", "H2c")
concl3 = ("incluir la centralidad MEJORA el modelo" if delta_aic > 2
          else "la centralidad no mejora significativamente el modelo (ΔAIC ≤ 2)")
P(f"Se ajustó un <b>ARIMAX{order}</b> para la <b>Demanda (Ener_1)</b> con exógenas "
  f"<b>Temperatura (Ener_3)</b> y la <b>centralidad de grado del nodo de origen</b>. "
  f"Comparando el AIC: sin centralidad = <b>{aic_base:.1f}</b>, con centralidad = "
  f"<b>{aic_full:.1f}</b> (ΔAIC = {delta_aic:.1f}). Conclusión: <b>{concl3}</b>. "
  "Cuando la mejora es material, la importancia topológica del nodo de despacho aporta "
  "poder explicativo que la temperatura por sí sola no captura: la <b>estructura de la red "
  "tiene valor predictivo</b> sobre la demanda y debe incorporarse a la planeación operativa.")
SP(0.2)

# --------------------------------------------------------------- Validation
P("Preguntas de Validación", "H1c")
P("1 · ¿Por qué no es válido Pearson directo sobre una serie con tendencia?", "H2c")
P("Pearson asume observaciones i.i.d. y estacionariedad. Dos series con tendencia (I(1)) "
  "comparten un componente temporal común y generan <b>correlación espuria</b> (miden que "
  "\"ambas crecen con el tiempo\", no una relación real); además la autocorrelación viola la "
  "independencia e <b>infla la significancia</b>. Correcto: diferenciar a I(0), probar "
  "cointegración o usar correlación de rango (Spearman).")

P("2 · Impacto del ruido de 5 dB en los coeficientes ARMA vs la versión clean", "H2c")
P("Con SNR bajo (~5 dB) el ruido blanco <b>atenúa los coeficientes AR hacia 0</b> "
  "(<i>attenuation bias</i>): la varianza extra diluye la autocorrelación estimada, el "
  "modelo subestima la persistencia y sobreestima la varianza del error/MA. La versión "
  "clean recupera coeficientes AR mayores y más estables (menor error estándar). Por eso el "
  "filtrado Butterworth (T4) es un preprocesamiento que mejora la identificación del ARMA.")

P("3 · ¿Cómo cambia la interpretación de un fallo si el sensor es un \"Bridge\"?", "H2c")
P("Un nodo puente (alta betweenness) es un punto de articulación: su caída no es un simple "
  "dato faltante local, sino que <b>fragmenta la red</b> en componentes y corta la "
  "observabilidad/despacho entre subredes. El fallo escala de \"medición perdida\" a "
  "\"<b>pérdida de conectividad sistémica</b>\"; su prioridad de mantenimiento y redundancia "
  "deben ser máximas.")

P("4 · ¿Cómo influye la posición geográfica en la varianza de la señal?", "H2c")
P("La geografía modula la varianza: en ladera/alta pendiente (proxy Agro_10) hay mayor "
  "turbulencia de viento, escorrentía y microclima, produciendo señales más ruidosas y de "
  "mayor varianza; en valle la señal es más estable. Esto explica el <b>agrupamiento "
  "espacial del bajo NDVI</b> (T1/P2): la calidad del dato depende del emplazamiento.")

SP(0.3)
P("Evidencia adicional de procesamiento de señales", "H2c")
add_figure("t3_ener4_spectrum.png",
           "Figura T3. Espectro FFT y PSD (Welch) de Ener_4: el ruido eleva las altas frecuencias.")
add_figure("t4_agro3_butterworth.png",
           "Figura T4. Agro_3 (RH): señal noise, referencia clean y filtrada Butterworth.")
add_figure("t2_ener5_rolling.png",
           "Figura T2. Costo del Gas (Ener_5): media y varianza móviles (ventana 50).")

SP(0.3)
P("Generado programáticamente con ReportLab a partir de los módulos de <i>src/</i> y las "
  "figuras de <i>figures/</i>. Reproducible con <i>py scripts/build_report.py</i>.", "Small")

# --------------------------------------------------------------- build
OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        leftMargin=2 * cm, rightMargin=2 * cm,
                        title="Informe Técnico — Challenge 02 TechLogistics")
doc.build(story)
print(f"PDF generado: {OUT}")
