"""
FIA - Actividad Parcial 2 - Sección 6 (2.6 Algoritmos genéticos) - PROYECTO FINAL EN EQUIPO
Caso empresarial (ilustrativo): una distribuidora regional reparte desde Amecameca a 13 comunidades y
regresa al origen. Se busca el orden de visita de MENOR distancia total (problema del agente viajero, TSP).
 
  Cromosoma : permutación de las 14 ciudades (orden de visita; ruta cerrada)
  Fitness   : longitud total de la ruta en km (minimizar), distancias ESTIMADAS = 1.3 x línea recta
  Operadores: selección por torneo (k=3), cruce ordenado OX (cxOrdered), mutación por intercambio de índices
  Población : 100 individuos · 40 generaciones (>= 30) · elitismo de 2 individuos
Referencias: óptimo exacto por programación dinámica (Held-Karp) y heurística del vecino más cercano.
 
Requisitos: pip install deap numpy matplotlib pillow
Uso:        python fia_s6_algoritmo_genetico.py   (genera figuras, evolucion_ruta.gif y resultados_s6.json)
"""
import json, random, time, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from deap import base, creator, tools, algorithms
from datos_edomex import COORDS, dist_km
 
CIUDADES = ["Amecameca", "Ozumba", "Atlautla", "Ecatzingo", "Tlalmanalco", "Chalco", "Ixtapaluca", "Juchitepec",
            "Tenango del Aire", "Cocotitlán", "Texcoco", "Cuautla", "Nezahualcóyotl", "Ciudad de México"]
N = len(CIUDADES)
D = np.array([[dist_km(a, b) for b in CIUDADES] for a in CIUDADES])
 
 
def longitud(ruta):
    """Longitud de la ruta cerrada (vuelve al inicio)."""
    return float(sum(D[ruta[i], ruta[(i + 1) % N]] for i in range(N)))
 
 
# ----------------------------------------------------------------------------- referencias
def held_karp():
    """Óptimo exacto (programación dinámica sobre subconjuntos), O(n^2 2^n); viable con n = 14."""
    INF = float("inf"); full = 1 << (N - 1)
    dp = [[INF] * (N - 1) for _ in range(full)]
    for j in range(N - 1): dp[1 << j][j] = D[N - 1][j]       # ciudad N-1 como origen auxiliar
    for m in range(full):
        for j in range(N - 1):
            v = dp[m][j]
            if v == INF or not (m >> j) & 1: continue
            for k in range(N - 1):
                if (m >> k) & 1: continue
                nm = m | (1 << k); c = v + D[j][k]
                if c < dp[nm][k]: dp[nm][k] = c
    return min(dp[full - 1][j] + D[j][N - 1] for j in range(N - 1))
 
 
def vecino_mas_cercano(inicio=0):
    ruta, rest = [inicio], set(range(N)) - {inicio}
    while rest:
        s = min(rest, key=lambda j: D[ruta[-1]][j]); ruta.append(s); rest.remove(s)
    return ruta, longitud(ruta)
 
 
# ----------------------------------------------------------------------------- algoritmo genético (DEAP)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)
 
 
def inversion(ind):
    i, j = sorted(random.sample(range(N), 2)); ind[i:j + 1] = reversed(ind[i:j + 1]); return (ind,)
 
 
def intercambio(ind):
    i, j = random.sample(range(N), 2); ind[i], ind[j] = ind[j], ind[i]; return (ind,)
 
 
def mutar_shuffle(ind):
    return tools.mutShuffleIndexes(ind, indpb=0.05)
 
 
CRUCES = {"OX": tools.cxOrdered, "PMX": tools.cxPartialyMatched}
MUTACIONES = {"intercambio de índices": mutar_shuffle, "inversión": inversion, "swap": intercambio}
 
 
def crear_toolbox(cruce="OX", mutacion="intercambio de índices", torneo=3):
    tb = base.Toolbox()
    tb.register("indices", random.sample, range(N), N)
    tb.register("individuo", tools.initIterate, creator.Individuo, tb.indices)
    tb.register("poblacion", tools.initRepeat, list, tb.individuo)
    tb.register("evaluate", lambda ind: (longitud(ind),))
    tb.register("mate", CRUCES[cruce]); tb.register("mutate", MUTACIONES[mutacion]); tb.register("select", tools.selTournament, tournsize=torneo)
    return tb
 
 
def correr_ga(semilla, n_pob=100, n_gen=40, cxpb=0.8, mutpb=0.2, elite=2, cruce="OX", mutacion="intercambio de índices", guardar_rutas=False):
    random.seed(semilla); np.random.seed(semilla); tb = crear_toolbox(cruce, mutacion)
    pop = tb.poblacion(n=n_pob); evals = 0
    for ind in pop: ind.fitness.values = tb.evaluate(ind)
    evals += len(pop); mejores, medias, rutas = [], [], []
    for g in range(n_gen + 1):
        fits = [i.fitness.values[0] for i in pop]; b = min(pop, key=lambda i: i.fitness.values[0])
        mejores.append(min(fits)); medias.append(float(np.mean(fits)))
        if guardar_rutas: rutas.append(list(b))
        if g == n_gen: break
        elites = [creator.Individuo(i) for i in tools.selBest(pop, elite)]
        for e, o in zip(elites, tools.selBest(pop, elite)): e.fitness.values = o.fitness.values
        hijos = algorithms.varAnd([creator.Individuo(i) for i in tb.select(pop, n_pob - elite)], tb, cxpb, mutpb)
        inval = [i for i in hijos if not i.fitness.valid]
        for ind in inval: ind.fitness.values = tb.evaluate(ind)
        evals += len(inval); pop = elites + hijos
    mejor = min(pop, key=lambda i: i.fitness.values[0])
    return dict(mejor=list(mejor), costo=mejor.fitness.values[0], mejores=mejores, medias=medias, evals=evals, rutas=rutas)
 
 
# ----------------------------------------------------------------------------- gráficas
VERDE = "#2E9E4F"
 
 
def fig_convergencia(runs, opt, nn, ruta="s6_convergencia.png"):
    M = np.array([r["mejores"] for r in runs]); A = np.array([r["medias"] for r in runs]); x = np.arange(M.shape[1])
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    ax.plot(x, M.mean(0), color=VERDE, lw=2.2, label="Mejor individuo (media de 10 corridas)"); ax.fill_between(x, M.min(0), M.max(0), color=VERDE, alpha=.15, label="Rango min–max")
    ax.plot(x, A.mean(0), color="#7A5C99", lw=1.8, label="Fitness promedio de la población")
    ax.axhline(opt, color="k", ls="--", lw=1, label=f"Óptimo exacto ({opt:.1f} km)"); ax.axhline(nn, color="#C2570C", ls=":", lw=1.4, label=f"Vecino más cercano ({nn:.1f} km)")
    ax.set_xlabel("Generación"); ax.set_ylabel("Longitud de la ruta (km)"); ax.set_title("Convergencia del algoritmo genético", fontweight="bold"); ax.legend(frameon=False, fontsize=7.5); ax.grid(alpha=.25)
    fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
 
 
def dibujar_ruta(ax, ruta, titulo=None, color=VERDE):
    xs = [COORDS[CIUDADES[i]][1] for i in ruta + [ruta[0]]]; ys = [COORDS[CIUDADES[i]][0] for i in ruta + [ruta[0]]]
    ax.plot(xs, ys, "-", color=color, lw=2, zorder=2)
    for k, i in enumerate(ruta):
        la, lo = COORDS[CIUDADES[i]]; ax.scatter(lo, la, s=60, c="gold" if i == 0 else "white", edgecolors="k", zorder=3)
        ax.annotate(CIUDADES[i], (lo, la), fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    if titulo: ax.set_title(titulo, fontsize=9, fontweight="bold")
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud"); ax.grid(alpha=.2)
 
 
def fig_rutas(mejor_ga, nn_ruta, ruta="s6_rutas.png"):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
    dibujar_ruta(axs[0], nn_ruta, f"Vecino más cercano: {longitud(nn_ruta):.1f} km", "#C2570C")
    dibujar_ruta(axs[1], mejor_ga, f"Algoritmo genético: {longitud(mejor_ga):.1f} km")
    fig.tight_layout(); fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
 
 
def fig_comparacion(vals, ruta="s6_comparacion.png"):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200); nombres = list(vals); cols = ["#999", "#C2570C", VERDE, "#222"]
    b = ax.bar(nombres, [vals[n] for n in nombres], color=cols[:len(nombres)])
    for r in b: ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 3, f"{r.get_height():.1f}", ha="center", fontsize=8)
    ax.set_ylabel("Longitud de la ruta (km)"); ax.set_title("Comparación de métodos", fontweight="bold"); ax.grid(alpha=.25, axis="y"); plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
 
 
def fig_ablacion(M, filas, cols, ruta="s6_ablacion.png"):
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200); im = ax.imshow(M, cmap="YlGn_r", aspect="auto")
    ax.set_xticks(range(len(cols)), cols, fontsize=8); ax.set_yticks(range(len(filas)), filas)
    for i in range(len(filas)):
        for j in range(len(cols)): ax.text(j, i, f"{M[i, j]:.1f} %", ha="center", va="center", fontsize=9)
    fig.colorbar(im, label="Brecha media vs. óptimo (%)"); ax.set_title("Plus: ablación de operadores (12 corridas)", fontweight="bold")
    fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
 
 
def gif_evolucion(rutas, opt, ruta="evolucion_ruta.gif", gens=(0, 1, 2, 3, 5, 8, 12, 18, 25, 32, 40)):
    frames = []
    for g in gens:
        fig, ax = plt.subplots(figsize=(6, 4.6), dpi=90); dibujar_ruta(ax, rutas[g], f"Generación {g}: {longitud(rutas[g]):.1f} km (óptimo {opt:.1f})")
        fig.tight_layout(); fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])); plt.close(fig)
    frames[0].save(ruta, save_all=True, append_images=frames[1:] + [frames[-1]] * 6, duration=550, loop=0)
 
 
def fig_tira(rutas, opt, ruta="s6_tira_evolucion.png", gens=(0, 3, 10, 40)):
    fig, axs = plt.subplots(1, len(gens), figsize=(12.5, 3.6), dpi=200)
    for ax, g in zip(axs, gens):
        dibujar_ruta(ax, rutas[g], f"Gen. {g}: {longitud(rutas[g]):.0f} km"); ax.set_xlabel(""); ax.set_ylabel(""); ax.set_xticks([]); ax.set_yticks([])
        for t in ax.texts: t.set_visible(False)
    fig.suptitle("Plus: evolución de la mejor ruta (cuadros de la animación)", fontsize=9); fig.tight_layout(); fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
 
 
if __name__ == "__main__":
    t = time.perf_counter(); opt = held_karp(); t_hk = time.perf_counter() - t
    nn_ruta, nn = vecino_mas_cercano(); rnd = np.mean([longitud(random.sample(range(N), N)) for _ in range(2000)])
    print(f"Óptimo exacto (Held-Karp): {opt:.2f} km ({t_hk:.1f} s) | Vecino más cercano: {nn:.2f} km (+{(nn / opt - 1) * 100:.1f} %) | Ruta aleatoria media: {rnd:.1f} km")
    runs = [correr_ga(s) for s in range(10)]
    costos = np.array([r["costo"] for r in runs]); evals = runs[0]["evals"]
    print(f"AG (pob=100, gen=40, cxpb=0.8, mutpb=0.2): mejor={costos.min():.2f} media={costos.mean():.2f} desv={costos.std():.2f} peor={costos.max():.2f} km | "
          f"brecha media={(costos.mean() / opt - 1) * 100:.2f} % | corridas óptimas={np.mean(costos <= opt + 1e-6):.0%} | evaluaciones={evals}")
    gen_opt = [next((g for g, v in enumerate(r["mejores"]) if v <= opt * 1.01), None) for r in runs]
    print("Generación en que se alcanza <=1 % del óptimo:", gen_opt)
    mejor_run = min(runs, key=lambda r: r["costo"]); print("Mejor ruta:", " > ".join(CIUDADES[i] for i in mejor_run["mejor"]))
    fig_convergencia(runs, opt, nn); fig_rutas(mejor_run["mejor"], nn_ruta); fig_comparacion({"Aleatoria (media)": rnd, "Vecino más cercano": nn, "Alg. genético (mejor)": costos.min(), "Óptimo exacto": opt})
    # PLUS: ablación de operadores y animación
    filas, cols = list(CRUCES), list(MUTACIONES); M = np.zeros((len(filas), len(cols)))
    for i, c in enumerate(filas):
        for j, m in enumerate(cols):
            M[i, j] = (np.mean([correr_ga(100 + s, cruce=c, mutacion=m)["costo"] for s in range(12)]) / opt - 1) * 100
    print("Ablación (brecha %, filas=cruce OX/PMX, columnas=mutación):\n", np.round(M, 2))
    fig_ablacion(M, filas, cols)
    g = correr_ga(mejor_run["costo"] and 3, guardar_rutas=True); gif_evolucion(g["rutas"], opt); fig_tira(g["rutas"], opt)
    json.dump(dict(opt=opt, nn=nn, rnd=float(rnd), costos=costos.tolist(), media=float(costos.mean()), desv=float(costos.std()), evals=evals,
                   exito=float(np.mean(costos <= opt + 1e-6)), gen_opt=gen_opt, mejor_ruta=[CIUDADES[i] for i in mejor_run["mejor"]], t_hk=t_hk,
                   ablacion=dict(filas=filas, cols=cols, M=M.tolist()), curva_media=np.mean([r["mejores"] for r in runs], 0).tolist(), ciudades=CIUDADES),
              open("resultados_s6.json", "w"), indent=1, ensure_ascii=False)
Salida real de la ejecución:
Óptimo exacto (Held-Karp): 268.33 km (0.2 s) | Vecino más cercano: 284.85 km (+6.2 %) | Ruta aleatoria media: 543.6 km
AG (pob=100, gen=40, cxpb=0.8, mutpb=0.2): mejor=268.33 media=272.52 desv=4.02 peor=279.19 km | brecha media=1.56 % | corridas óptimas=10% | evaluaciones=4020
Generación en que se alcanza <=1 % del óptimo: [32, 27, None, 15, None, 11, None, 24, None, 12]
Mejor ruta: Ecatzingo > Atlautla > Ozumba > Amecameca > Tlalmanalco > Cocotitlán > Ixtapaluca > Texcoco > Ciudad de México > Nezahualcóyotl > Chalco > Tenango del Aire > Juchitepec > Cuautla
Ablación (brecha %, filas=cruce OX/PMX, columnas=mutación):
 [[1.97 0.19 0.24]
 [3.76 1.05 3.63]]
