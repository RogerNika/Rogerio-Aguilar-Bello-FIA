"""
FIA - Actividad Parcial 2 - Sección 2 (2.1 Problemas de búsqueda, 2.2 Espacios de búsqueda)
 
Modela DOS problemas como problemas de búsqueda (estado inicial, acciones,
modelo de transición, costo y prueba de meta), construye su espacio de
estados como grafo dirigido y ponderado con NetworkX, genera los diagramas
y calcula la complejidad del espacio.
 
  Problema A: ruteo de una unidad de reparto (visitar todas las paradas y regresar).
  Problema B: robot de almacén en una cuadrícula con obstáculos y zonas lentas.
 
Requisitos: pip install networkx matplotlib
Uso:        python fia_s2_espacios_busqueda.py
"""
from itertools import count
from math import factorial
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
 
# ===========================================================================
# PROBLEMA A: RUTEO DE REPARTO
# Estado   : (ubicacion_actual, paradas_visitadas)  -> paradas_visitadas es frozenset
# Acciones : ir a una parada aun no visitada; con todas visitadas, volver al depósito
# Costo    : minutos de viaje entre ubicaciones (DATOS ILUSTRATIVOS, no medidos)
# Meta     : (depósito, todas las paradas visitadas)
# ===========================================================================
DEPOSITO = "D"
PARADAS = ["A", "B", "C", "E"]  # 4 clientes
TIEMPOS = {  # minutos, simétricos; reemplazar con datos reales (API de mapas / GPS)
    ("D", "A"): 12, ("D", "B"): 20, ("D", "C"): 25, ("D", "E"): 15,
    ("A", "B"): 10, ("A", "C"): 18, ("A", "E"): 14,
    ("B", "C"): 9,  ("B", "E"): 22,
    ("C", "E"): 11,
}
 
 
def tiempo(u, v):
    return TIEMPOS.get((u, v), TIEMPOS.get((v, u)))
 
 
def estado_inicial_A():
    return (DEPOSITO, frozenset())
 
 
def es_meta_A(s):
    loc, vis = s
    return loc == DEPOSITO and len(vis) == len(PARADAS)
 
 
def acciones_A(s):
    """Generador de sucesores: devuelve (accion, nuevo_estado, costo)."""
    loc, vis = s
    pendientes = [p for p in PARADAS if p not in vis]
    if pendientes:
        for p in pendientes:
            yield (f"ir a {p}", (p, vis | {p}), tiempo(loc, p))
    elif loc != DEPOSITO:
        yield ("volver a D", (DEPOSITO, vis), tiempo(loc, DEPOSITO))
 
 
def generar_espacio(estado_ini, sucesores):
    """Recorre (BFS) todos los estados alcanzables y arma el grafo dirigido ponderado."""
    G = nx.DiGraph()
    G.add_node(estado_ini)
    frontera = [estado_ini]
    while frontera:
        s = frontera.pop(0)
        for accion, s2, costo in sucesores(s):
            if s2 not in G:
                G.add_node(s2)
                frontera.append(s2)
            G.add_edge(s, s2, weight=costo, accion=accion)
    return G
 
 
def etiqueta_A(s):
    loc, vis = s
    return f"{loc}\n{{{''.join(sorted(vis))}}}"
 
 
# ===========================================================================
# PROBLEMA B: ROBOT DE ALMACÉN
# Estado   : (fila, columna)
# Acciones : arriba, abajo, izquierda, derecha (si la celda existe y no es muro)
# Costo    : 1 en pasillo normal; 3 en zona lenta (piso mojado/carga)
# Meta     : celda de entrega
# ===========================================================================
MAPA = [
    "S.#...",
    ".##.#.",
    "....~.",
    ".#.#~.",
    ".#...G",
]  # S=inicio, G=meta, #=muro, ~=zona lenta (costo 3), .=pasillo (costo 1)
MOV = {"arriba": (-1, 0), "abajo": (1, 0), "izquierda": (0, -1), "derecha": (0, 1)}
 
 
def buscar(simbolo):
    for r, fila in enumerate(MAPA):
        if simbolo in fila:
            return (r, fila.index(simbolo))
 
 
INICIO_B, META_B = buscar("S"), buscar("G")
 
 
def es_meta_B(s):
    return s == META_B
 
 
def acciones_B(s):
    r, c = s
    for nombre, (dr, dc) in MOV.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(MAPA) and 0 <= nc < len(MAPA[0]) and MAPA[nr][nc] != "#":
            yield (nombre, (nr, nc), 3 if MAPA[nr][nc] == "~" else 1)
 
 
# ===========================================================================
# COMPLEJIDAD
# ===========================================================================
def analisis(G, sucesores, nombre, es_meta, estado_ini):
    """Mide |S|, |A|, factor de ramificación b y profundidad máxima del grafo."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    b_prom = m / n
    b_max = max(d for _, d in G.out_degree())
    metas = [s for s in G if es_meta(s)]
    prof = nx.single_source_shortest_path_length(G, estado_ini)
    d_meta = min(prof[s] for s in metas)
    d_max = max(prof.values())
    print(f"[{nombre}] estados={n} transiciones={m} b_prom={b_prom:.2f} b_max={b_max} "
          f"profundidad_meta={d_meta} profundidad_max={d_max} estados_meta={len(metas)}")
    return dict(n=n, m=m, b_prom=b_prom, b_max=b_max, d_meta=d_meta, d_max=d_max)
 
 
def estados_A_formula(k):
    """|S| teórico: inicio + (parada_actual, subconjunto con ella) + estado meta."""
    return 1 + k * 2 ** (k - 1) + 1
 
 
def tabla_escalamiento():
    print("\n k  | estados (grafo) | secuencias (arbol, k!) | razon")
    for k in (4, 8, 12, 16, 20):
        s, a = estados_A_formula(k), factorial(k)
        print(f"{k:>3} | {s:>15,} | {a:>22,} | {a / s:,.1f}x")
 
 
# ===========================================================================
# DIAGRAMAS
# ===========================================================================
def dibujar_A(G, ruta="grafo_problema_A.png"):
    # Capas por número de paradas visitadas (nivel) para ordenar el dibujo
    niveles = {}
    for s in G:
        niv = len(s[1]) + (1 if es_meta_A(s) else 0)
        niveles.setdefault(niv, []).append(s)
    pos = {}
    for niv, nodos in niveles.items():
        nodos.sort(key=lambda s: (s[0], sorted(s[1])))
        for i, s in enumerate(nodos):
            pos[s] = (niv * 2.0, (len(nodos) - 1) / 2 * 1.35 - i * 1.35)
    colores = ["#0F7B6C" if s == estado_inicial_A() else "#C2570C" if es_meta_A(s) else "#DDE3F5" for s in G]
    fig, ax = plt.subplots(figsize=(10, 11), dpi=170)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#8A93B5", arrows=True, arrowsize=9, width=0.7,
                           connectionstyle="arc3,rad=0.05", node_size=1500)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colores, node_size=1500, edgecolors="#1F2A5C")
    nx.draw_networkx_labels(G, pos, {s: etiqueta_A(s) for s in G}, font_size=7, ax=ax)
    ax.set_title("Problema A: espacio de estados del ruteo de reparto (4 paradas)\n"
                 "Verde = inicio · Naranja = meta · Nodo = (ubicación, {visitadas})\n"
                 "Cada columna = nº de paradas visitadas", fontsize=10, color="#1F2A5C")
    ax.axis("off")
    fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
 
 
def dibujar_B(G, ruta="grafo_problema_B.png"):
    pos = {s: (s[1], -s[0]) for s in G}  # la celda (fila, col) se dibuja donde está
    colores = ["#0F7B6C" if s == INICIO_B else "#C2570C" if s == META_B
               else "#F2D9A7" if MAPA[s[0]][s[1]] == "~" else "#DDE3F5" for s in G]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=170)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#8A93B5", arrows=True, arrowsize=8, width=0.6,
                           connectionstyle="arc3,rad=0.12", node_size=700)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colores, node_size=700, edgecolors="#1F2A5C")
    nx.draw_networkx_labels(G, pos, {s: f"{s[0]},{s[1]}" for s in G}, font_size=6, ax=ax)
    for r, fila in enumerate(MAPA):  # muros dibujados como cuadros grises
        for c, ch in enumerate(fila):
            if ch == "#":
                ax.add_patch(plt.Rectangle((c - .4, -r - .4), .8, .8, color="#9AA0B4"))
    ax.set_title("Problema B: espacio de estados del robot de almacén (grafo de celdas)\n"
                 "Verde = inicio · Naranja = meta · Beige = zona lenta (costo 3) · Gris = muro",
                 fontsize=9, color="#1F2A5C")
    ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(ruta, bbox_inches="tight"); plt.close(fig)
 
 
# ===========================================================================
# EJECUCIÓN
# ===========================================================================
if __name__ == "__main__":
    GA = generar_espacio(estado_inicial_A(), acciones_A)
    GB = generar_espacio(INICIO_B, acciones_B)
 
    ra = analisis(GA, acciones_A, "A reparto", es_meta_A, estado_inicial_A())
    rb = analisis(GB, acciones_B, "B almacén", es_meta_B, INICIO_B)
    print(f"Verificación A: estados reales={ra['n']} vs fórmula 1+k*2^(k-1)+1={estados_A_formula(len(PARADAS))}")
    celdas_libres = sum(f.count(".") + f.count("~") + f.count("S") + f.count("G") for f in MAPA)
    print(f"Verificación B: estados={rb['n']} vs celdas libres={celdas_libres}")
    tabla_escalamiento()
 
    # Solución óptima (costo uniforme con Dijkstra de NetworkX) para validar el modelo
    metaA = next(s for s in GA if es_meta_A(s))
    rutaA = nx.dijkstra_path(GA, estado_inicial_A(), metaA, weight="weight")
    costoA = nx.dijkstra_path_length(GA, estado_inicial_A(), metaA, weight="weight")
    print("\nRuta óptima A:", " -> ".join(s[0] for s in rutaA), f"| costo = {costoA} min")
    rutaB = nx.dijkstra_path(GB, INICIO_B, META_B, weight="weight")
    costoB = nx.dijkstra_path_length(GB, INICIO_B, META_B, weight="weight")
    print("Ruta óptima B:", rutaB, f"| costo = {costoB}")
 
    dibujar_A(GA); dibujar_B(GB)
    print("Diagramas: grafo_problema_A.png, grafo_problema_B.png")
