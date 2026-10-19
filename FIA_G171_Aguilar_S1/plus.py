"""
PLUS Sección 1: Escala de Evidencia de IA (EEIA), rúbrica propia de 5 criterios (0-2 pts c/u, máx. 10).
Evalúa qué tan sólida es la evidencia publicada de cada caso. Es un criterio SUBJETIVO del alumno,
con justificación por cada puntaje. Genera: eeia_radar.png y eeia_puntajes.csv
Requisitos: pip install matplotlib pandas
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
CRITERIOS = ["Revisión editorial", "Método descrito", "KPI cuantificado", "Evaluación independiente", "Replicabilidad"]
# 0 = no cumple · 1 = cumple parcialmente · 2 = cumple
PUNTAJES = {
    "IDx-DR (salud)":        [2, 2, 2, 1, 1],
    "Google/DeepMind (industria)": [0, 1, 1, 0, 0],
    "UPS ORION (movilidad)": [2, 1, 1, 1, 0],
}
COLORES = ["#0F7B6C", "#C2570C", "#1F2A5C"]
 
 
def tabla():
    df = pd.DataFrame(PUNTAJES, index=CRITERIOS).T
    df["Total /10"] = df.sum(axis=1)
    return df
 
 
def radar(ruta="eeia_radar.png"):
    ang = np.linspace(0, 2 * np.pi, len(CRITERIOS), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=200, subplot_kw=dict(polar=True))
    for (caso, v), col in zip(PUNTAJES.items(), COLORES):
        vals = v + v[:1]
        ax.plot(ang, vals, color=col, lw=2, label=f"{caso}: {sum(v)}/10")
        ax.fill(ang, vals, color=col, alpha=0.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([c.replace(" ", "\n", 1) for c in CRITERIOS], fontsize=8)
    ax.set_yticks([0, 1, 2]); ax.set_ylim(0, 2); ax.set_yticklabels(["0", "1", "2"], fontsize=7)
    ax.set_title("Escala de Evidencia de IA (EEIA)", fontsize=12, fontweight="bold", pad=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=8, frameon=False)
    fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
 
 
if __name__ == "__main__":
    df = tabla()
    print(df.to_string())
    df.to_csv("eeia_puntajes.csv", encoding="utf-8-sig")
    radar()
    print("Generado: eeia_radar.png, eeia_puntajes.csv")
