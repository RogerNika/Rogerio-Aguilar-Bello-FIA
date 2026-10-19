"""
FIA - Actividad Parcial 2 - Sección 1 (1.7 Aplicaciones y casos de estudio de IA)
Genera: (1) matriz comparativa en CSV, (2) métricas de verificación, (3) infografía PNG.

Requisitos: pip install pandas matplotlib
Uso:        python fia_s1_casos_ia.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# 1. DATOS: cada caso con su fuente verificable (ver Referencias APA 7 en el PDF)
# ---------------------------------------------------------------------------
CASOS = [
    {
        "caso": "IDx-DR (diagnóstico autónomo)",
        "sector": "Salud",
        "pais": "EE. UU.",
        "rama_ia": "Aprendizaje profundo / visión por computadora",
        "tecnica": "Redes neuronales sobre imágenes de fondo de ojo",
        "kpi_principal": "Sensibilidad 87.2 % | Especificidad 90.7 %",
        "kpi_valor": "87.2 %",
        "kpi_etiqueta": "sensibilidad",
        "fuente": "Abràmoff et al. (2018)",
        "anio": 2018,
    },
    {
        "caso": "Enfriamiento de centros de datos de Google",
        "sector": "Industria",
        "pais": "EE. UU. / Reino Unido",
        "rama_ia": "Aprendizaje automático supervisado",
        "tecnica": "Redes neuronales que predicen PUE y temperatura",
        "kpi_principal": "-40 % energía de enfriamiento | -15 % sobrecarga PUE",
        "kpi_valor": "-40 %",
        "kpi_etiqueta": "energía de enfriamiento",
        "fuente": "Evans y Gao (2016)",
        "anio": 2016,
    },
    {
        "caso": "UPS ORION (ruteo de entregas)",
        "sector": "Movilidad / logística",
        "pais": "EE. UU.",
        "rama_ia": "Búsqueda y optimización heurística",
        "tecnica": "Metaheurística de búsqueda local + telemática",
        "kpi_principal": "100 M millas y 10 M galones menos por año",
        "kpi_valor": "100 M",
        "kpi_etiqueta": "millas menos por año",
        "fuente": "Holland et al. (2017)",
        "anio": 2017,
    },
]


# ---------------------------------------------------------------------------
# 2. MATRIZ COMPARATIVA
# ---------------------------------------------------------------------------
def construir_matriz(casos):
    cols = ["caso", "rama_ia", "tecnica", "sector", "kpi_principal", "pais", "fuente", "anio"]
    return pd.DataFrame(casos)[cols]


# ---------------------------------------------------------------------------
# 3. MÉTRICAS DE VERIFICACIÓN (comprueban la coherencia de las cifras publicadas)
# ---------------------------------------------------------------------------
def ppv_npv(sens, spec, prevalencia):
    """Valores predictivos positivo/negativo (Bayes) para una prevalencia dada."""
    tp = sens * prevalencia
    fn = (1 - sens) * prevalencia
    tn = spec * (1 - prevalencia)
    fp = (1 - spec) * (1 - prevalencia)
    return tp / (tp + fp), tn / (tn + fn)


def millas_por_conductor_dia(millas_anuales, conductores, dias_laborales=250):
    """Ahorro diario por conductor. dias_laborales es SUPUESTO del alumno."""
    return millas_anuales / conductores / dias_laborales


def verificaciones():
    out = {}
    # IDx-DR: prevalencia 10 % es HIPOTÉTICA (ilustra el efecto de la prevalencia)
    ppv, npv = ppv_npv(0.872, 0.907, 0.10)
    out["IDx-DR PPV (prev. 10 %)"] = f"{ppv:.1%}"
    out["IDx-DR NPV (prev. 10 %)"] = f"{npv:.1%}"
    # ORION: 100 M millas / 55,000 conductores / 250 días
    out["ORION millas/conductor/día"] = f"{millas_por_conductor_dia(100e6, 55_000):.1f}"
    # Google: consistencia -40 % enfriamiento -> -15 % sobrecarga PUE (cifras publicadas)
    out["Google: ratio PUE/enfriamiento"] = f"{15 / 40:.3f}"
    return out


# ---------------------------------------------------------------------------
# 4. INFOGRAFÍA COMPARATIVA
# ---------------------------------------------------------------------------
def infografia(casos, ruta="infografia_comparativa.png"):
    """Infografía tipo ruta: cada caso es un carril con 5 etapas (problema -> datos -> técnica -> decisión -> impacto)."""
    from matplotlib.patches import Polygon
    colores = ["#0F7B6C", "#C2570C", "#1F2A5C"]
    etapas = ["1  PROBLEMA", "2  DATOS", "3  TÉCNICA DE IA", "4  DECISIÓN", "5  IMPACTO MEDIBLE"]
    carriles = [
        ["Tamizaje de retinopatía diabética que depende de especialistas",
         "Imágenes de fondo de ojo (900 pacientes, 10 consultorios)",
         "Aprendizaje profundo y visión por computadora",
         "Referir o no al paciente con un especialista",
         "87.2 %|sensibilidad · 90.7 % especificidad"],
        ["El enfriamiento del centro de datos consume mucha energía",
         "Datos históricos de sensores de la instalación",
         "Redes neuronales (aprendizaje supervisado)",
         "Ajustar el enfriamiento según el PUE y la temperatura predichos",
         "-40 %|energía de enfriamiento"],
        ["Ruta diaria de cada uno de los 55,000 conductores",
         "Paquetes del día, datos telemáticos y de direcciones",
         "Metaheurística (búsqueda y optimización heurística)",
         "Ruta optimizada y consistente de un día a otro",
         "100 M|millas menos por año"],
    ]
    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=200)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(2, 96, "De la necesidad al impacto: la ruta de tres casos reales de IA", fontsize=14.5,
            fontweight="bold", color="#1F2A5C", va="center")
    ax.text(2, 91.6, "Salud · Industria · Movilidad", fontsize=9.5, color="#666666", va="center")
    x_ini, ancho, punta = 13.5, 16.9, 2.6
    for i, e in enumerate(etapas):                       # encabezados de etapa
        ax.text(x_ini + i * ancho + ancho / 2, 86.2, e, ha="center", va="center", fontsize=7.4,
                fontweight="bold", color="#444444")
    for j, (c, col, textos) in enumerate(zip(casos, colores, carriles)):
        y0 = 62 - j * 26; alto = 20
        ax.add_patch(plt.Rectangle((1.5, y0), 10.5, alto, fc=col, ec="none"))   # etiqueta del carril
        ax.text(6.75, y0 + alto * 0.68, c["sector"].split(" /")[0].upper(), ha="center", va="center",
                color="white", fontsize=8.6, fontweight="bold")
        ax.text(6.75, y0 + alto * 0.36, _envolver(c["caso"].split(" (")[0], 13), ha="center", va="center",
                color="white", fontsize=6.6)
        for i, t in enumerate(textos):
            x = x_ini + i * ancho
            pts = [(x, y0), (x + ancho - punta, y0), (x + ancho, y0 + alto / 2), (x + ancho - punta, y0 + alto), (x, y0 + alto)]
            if i > 0:
                pts.append((x + punta, y0 + alto / 2))
            ultimo = i == len(textos) - 1
            ax.add_patch(Polygon(pts, closed=True, fc=col if ultimo else "#F4F5FA", ec=col, lw=1.4,
                                 alpha=1 if ultimo else 0.55 + 0.1 * i))
            cx, cy = x + ancho / 2 + (0.6 if i else -0.2), y0 + alto / 2
            if ultimo:
                num, lab = t.split("|")
                ax.text(cx, cy + 3.6, num, ha="center", va="center", color="white", fontsize=22, fontweight="bold")
                ax.text(cx, cy - 4.2, _envolver(lab, 17), ha="center", va="center", color="white", fontsize=6.8)
            else:
                ax.text(cx, cy, _envolver(t, 20), ha="center", va="center", color="#222222", fontsize=6.8)
    ax.text(2, 3.0, "Fuentes: Abràmoff et al. (2018) · Evans y Gao (2016) · Holland et al. (2017)", fontsize=7,
            color="#666666", va="center")
    ax.text(98, 3.0, "Elaboración propia", fontsize=7, color="#666666", va="center", ha="right")
    fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
    return ruta


def _envolver(texto, n):
    palabras, linea, lineas = texto.split(), "", []
    for p in palabras:
        if len(linea) + len(p) + 1 > n:
            lineas.append(linea)
            linea = p
        else:
            linea = (linea + " " + p).strip()
    lineas.append(linea)
    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# 5. EJECUCIÓN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    matriz = construir_matriz(CASOS)
    matriz.to_csv("matriz_casos_ia.csv", index=False, encoding="utf-8-sig")
    print(matriz.to_string(index=False))
    print("\n--- Verificaciones ---")
    for k, v in verificaciones().items():
        print(f"{k}: {v}")
    print("\nInfografía:", infografia(CASOS))