"""
FIA - Actividad Parcial 2 - Sección 4 (2.4 Búsqueda ciega: BFS / DFS)
Resuelve laberintos en cuadrícula con Búsqueda en Anchura (BFS) y en Profundidad (DFS) y los compara.
 
  Grafo      : cada celda libre es un nodo; aristas entre celdas libres adyacentes (lista de adyacencia)
  Estado     : celda (fila, col)      Acciones: arriba, derecha, abajo, izquierda      Costo: 1 por paso
  Inicio/Meta: esquina superior izquierda / inferior derecha
  Métricas   : nodos expandidos, tamaño máximo de la frontera (memoria), longitud del camino, tiempo
 
Requisitos: pip install numpy matplotlib pillow
Uso:        python fia_s4_bfs_dfs.py   (genera figuras, animacion_bfs_dfs.gif y resultados_s4.json)
"""
import json, random, time
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
 
ORDEN = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # arriba, derecha, abajo, izquierda (orden fijo de expansión)
 
 
# ----------------------------------------------------------------------------- laberinto
def generar_laberinto(n, semilla, prob_ciclos=0.0):
    """n impar. 1 = muro, 0 = libre. Retroceso recursivo (iterativo) => laberinto perfecto (camino único).
    prob_ciclos > 0 abre muros extra y crea caminos alternativos ("laberinto con ciclos")."""
    rnd = random.Random(semilla); g = np.ones((n, n), dtype=np.uint8); g[1, 1] = 0; pila = [(1, 1)]
    while pila:
        r, c = pila[-1]
        vec = [(r + 2 * dr, c + 2 * dc, dr, dc) for dr, dc in ORDEN if 0 < r + 2 * dr < n - 1 and 0 < c + 2 * dc < n - 1 and g[r + 2 * dr, c + 2 * dc]]
        if vec:
            nr, nc, dr, dc = rnd.choice(vec); g[r + dr, c + dc] = 0; g[nr, nc] = 0; pila.append((nr, nc))
        else:
            pila.pop()
    if prob_ciclos > 0:
        for r in range(1, n - 1):
            for c in range(1, n - 1):
                if g[r, c] and rnd.random() < prob_ciclos and ((g[r - 1, c] == 0 and g[r + 1, c] == 0) or (g[r, c - 1] == 0 and g[r, c + 1] == 0)):
                    g[r, c] = 0
    return g
 
 
def vecinos(g, nodo):
    r, c = nodo; n = g.shape[0]
    for dr, dc in ORDEN:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n and g[nr, nc] == 0:
            yield (nr, nc)
 
 
def camino(padres, meta):
    p, nodo = [], meta
    while nodo is not None:
        p.append(nodo); nodo = padres[nodo]
    return p[::-1]
 
 
# ----------------------------------------------------------------------------- algoritmos
def bfs(g, ini, meta, registrar=False):
    """BFS: cola FIFO. Marca visitado al ENCOLAR. Óptimo en número de pasos (costo uniforme)."""
    t0 = time.perf_counter(); frontera = deque([ini]); padres = {ini: None}; orden = []; max_f = 1
    while frontera:
        max_f = max(max_f, len(frontera)); nodo = frontera.popleft(); orden.append(nodo)
        if nodo == meta:
            break
        for v in vecinos(g, nodo):
            if v not in padres:
                padres[v] = nodo; frontera.append(v)
    ok = meta in padres and orden[-1] == meta
    return dict(ok=ok, camino=camino(padres, meta) if ok else None, expandidos=len(orden), max_frontera=max_f,
                tiempo=time.perf_counter() - t0, orden=orden if registrar else None)
 
 
def dfs(g, ini, meta, registrar=False):
    """DFS: pila LIFO. Marca visitado al DESAPILAR. No garantiza el camino más corto."""
    t0 = time.perf_counter(); pila = [(ini, None)]; padres = {}; orden = []; max_f = 1
    while pila:
        max_f = max(max_f, len(pila)); nodo, padre = pila.pop()
        if nodo in padres:
            continue
        padres[nodo] = padre; orden.append(nodo)
        if nodo == meta:
            break
        for v in reversed(list(vecinos(g, nodo))):   # reversed => se explora primero "arriba"
            if v not in padres:
                pila.append((v, nodo))
    ok = meta in padres
    return dict(ok=ok, camino=camino(padres, meta) if ok else None, expandidos=len(orden), max_frontera=max_f,
                tiempo=time.perf_counter() - t0, orden=orden if registrar else None)
 
 
# ----------------------------------------------------------------------------- experimentos
def experimento(tamanos=(15, 31, 51, 71), semillas=range(10)):
    filas = []
    for tipo, pc in (("perfecto", 0.0), ("con ciclos", 0.10)):
        for n in tamanos:
            acum = {"BFS": [], "DFS": []}
            for s in semillas:
                g = generar_laberinto(n, s, pc); ini, meta = (1, 1), (n - 2, n - 2)
                for nombre, fn in (("BFS", bfs), ("DFS", dfs)):
                    r = fn(g, ini, meta); assert r["ok"]
                    r["tiempo"] = min(fn(g, ini, meta)["tiempo"] for _ in range(3))
                    acum[nombre].append(r)
            for nombre in acum:
                R = acum[nombre]
                filas.append(dict(tipo=tipo, n=n, algoritmo=nombre, libres=int(np.mean([(generar_laberinto(n, s, pc) == 0).sum() for s in semillas])),
                                  expandidos=float(np.mean([r["expandidos"] for r in R])), max_frontera=float(np.mean([r["max_frontera"] for r in R])),
                                  largo=float(np.mean([len(r["camino"]) - 1 for r in R])), tiempo_ms=float(np.mean([r["tiempo"] for r in R]) * 1000)))
    return filas
 
 
# ----------------------------------------------------------------------------- gráficas
COL = {"BFS": "#1F6FEB", "DFS": "#E0662B"}
 
 
def fig_barras(filas, ruta="s4_metricas.png"):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), dpi=200)
    ncon = [f for f in filas if f["tipo"] == "con ciclos"]; ns = sorted({f["n"] for f in ncon}); x = np.arange(len(ns))
    for ax, key, tit in zip(axs, ("expandidos", "max_frontera", "largo"), ("Nodos expandidos", "Frontera máxima (memoria)", "Longitud del camino (pasos)")):
        for i, alg in enumerate(("BFS", "DFS")):
            ax.bar(x + (i - .5) * .38, [next(f[key] for f in ncon if f["n"] == n and f["algoritmo"] == alg) for n in ns], .38, color=COL[alg], label=alg)
        ax.set_xticks(x, [f"{n}×{n}" for n in ns]); ax.set_title(tit, fontsize=9, fontweight="bold"); ax.grid(alpha=.25, axis="y")
        if key != "largo": ax.set_yscale("log")
    axs[0].legend(frameon=False); fig.suptitle("Laberintos con ciclos (promedio de 10 semillas)", fontsize=10)
    fig.tight_layout(); fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
 
 
def pintar(g, visitados, cam=None, ini=None, meta=None, esc=1):
    n = g.shape[0]; img = np.full((n, n, 3), 255, np.uint8); img[g == 1] = (30, 30, 40)
    for k, (r, c) in enumerate(visitados):
        img[r, c] = (200, 215, 245)
    return img
 
 
def fig_solucion(g, rb, rd, ruta="s4_soluciones.png"):
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.7), dpi=200)
    for ax, r, alg in zip(axs, (rb, rd), ("BFS", "DFS")):
        n = g.shape[0]; img = np.full((n, n, 3), 1.0, float); img[g == 1] = (.12, .12, .16)
        orden = r["orden"]; cmap = plt.get_cmap("Blues" if alg == "BFS" else "Oranges")
        for k, (a, b) in enumerate(orden): img[a, b] = cmap(.25 + .5 * k / len(orden))[:3]
        for a, b in r["camino"]: img[a, b] = (.1, .6, .2)
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{alg}: {r['expandidos']} nodos expandidos · camino de {len(r['camino']) - 1} pasos", fontsize=8, fontweight="bold", color=COL[alg])
    fig.suptitle("Laberinto 31×31 con ciclos: celdas exploradas (azul/naranja) y camino solución (verde)", fontsize=9)
    fig.tight_layout(); fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
 
 
# ----------------------------------------------------------------------------- PLUS: animación
def render(g, orden, k, cam, color, escala=12):
    n = g.shape[0]; img = np.full((n, n, 3), 255, np.uint8); img[g == 1] = (30, 30, 40)
    for (a, b) in orden[:k]: img[a, b] = color
    if k >= len(orden) and cam:
        for (a, b) in cam: img[a, b] = (30, 170, 70)
    elif k > 0:
        a, b = orden[k - 1]; img[a, b] = (220, 30, 30)
    img[1, 1] = (0, 0, 0) if g[1, 1] == 0 else img[1, 1]
    return Image.fromarray(img).resize((n * escala, n * escala), Image.NEAREST)
 
 
def animacion(g, rb, rd, ruta="animacion_bfs_dfs.gif", pasos=70):
    n = g.shape[0]; E = 12; frames = []; maxk = max(len(rb["orden"]), len(rd["orden"]))
    ks = sorted(set(int(round(i * maxk / pasos)) for i in range(pasos + 1))) + [maxk] * 8
    for k in ks:
        kb, kd = min(k, len(rb["orden"])), min(k, len(rd["orden"]))
        a = render(g, rb["orden"], kb, rb["camino"], (120, 160, 235), E); b = render(g, rd["orden"], kd, rd["camino"], (240, 160, 110), E)
        canvas = Image.new("RGB", (2 * n * E + 12, n * E + 26), "white"); canvas.paste(a, (0, 26)); canvas.paste(b, (n * E + 12, 26))
        d = ImageDraw.Draw(canvas); d.text((6, 6), f"BFS  expandidos: {kb}", fill=(31, 111, 235)); d.text((n * E + 18, 6), f"DFS  expandidos: {kd}", fill=(224, 102, 43))
        frames.append(canvas)
    frames[0].save(ruta, save_all=True, append_images=frames[1:], duration=90, loop=0)
    return frames
 
 
def tira(g, rb, rd, ruta="s4_tira_animacion.png", cortes=(0.1, 0.35, 0.7, 1.0)):
    fig, axs = plt.subplots(2, len(cortes), figsize=(9.2, 4.9), dpi=200)
    for j, q in enumerate(cortes):
        for i, (r, col, alg) in enumerate(((rb, (120, 160, 235), "BFS"), (rd, (240, 160, 110), "DFS"))):
            k = max(1, int(len(r["orden"]) * q)); im = render(g, r["orden"], k, r["camino"], col, 8)
            axs[i, j].imshow(im); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            axs[i, j].set_title(f"{alg}: {k} nodos", fontsize=7, color=COL[alg], fontweight="bold")
    fig.suptitle("Plus: la exploración paso a paso (cuadros de la animación)", fontsize=9); fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
 
 
if __name__ == "__main__":
    filas = experimento()
    print(f"{'tipo':11s} {'n':>3s} {'alg':4s} {'libres':>6s} {'expand.':>9s} {'frontera':>9s} {'largo':>7s} {'t(ms)':>8s}")
    for f in filas:
        print(f"{f['tipo']:11s} {f['n']:>3d} {f['algoritmo']:4s} {f['libres']:>6d} {f['expandidos']:>9.0f} {f['max_frontera']:>9.0f} {f['largo']:>7.1f} {f['tiempo_ms']:>8.2f}")
    g = generar_laberinto(31, 3, 0.10); rb = bfs(g, (1, 1), (29, 29), True); rd = dfs(g, (1, 1), (29, 29), True)
    print(f"\nEjemplo 31x31 con ciclos: BFS exp={rb['expandidos']} camino={len(rb['camino'])-1} | DFS exp={rd['expandidos']} camino={len(rd['camino'])-1}")
    fig_barras(filas); fig_solucion(g, rb, rd)
    g2 = generar_laberinto(21, 5, 0.10); b2 = bfs(g2, (1, 1), (19, 19), True); d2 = dfs(g2, (1, 1), (19, 19), True)
    animacion(g2, b2, d2); tira(g2, b2, d2)
    json.dump(dict(filas=filas, ejemplo=dict(bfs_exp=rb["expandidos"], dfs_exp=rd["expandidos"], bfs_largo=len(rb["camino"]) - 1, dfs_largo=len(rd["camino"]) - 1)),
              open("resultados_s4.json", "w"), indent=1)
