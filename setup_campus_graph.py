import math
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

CENTER = (-7.2819, 112.7948)
RADIUS = 1500

STOP_PATTERNS = [
    "^Departemen Teknik Informatika$",
    "Teknik Komputer",
    "Teknik Kimia ITS",
    "Tower 2",
    "Tribun Stadion",
    "Departemen Desain Interior",
    "Departemen Desain Produk",
    "^Perpustakaan ITS$",
    "^REKTORAT Institut",
    "Manarul Ilmi",
    "Teknik Elektro ITS",
    "^Teknik Mesin$",
    "^Teknik Sipil$",
    "Teknik Industri",
    "Teknik Fisika",
    "^Matematika$",
    "^Statistika$",
    "^Biologi$",
]

MANUAL_STOPS = [
    {"name": "Teknik Informatika ITS", "lat": None, "lon": None},
]

Path("data").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

G = ox.graph_from_point(CENTER, dist=RADIUS, network_type="walk", simplify=True)
G = G.to_undirected()
G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
ox.save_graphml(G, "data/its_walk.graphml")
print("nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())

bld = ox.features_from_point(CENTER, tags={"building": True}, dist=RADIUS)
bld = bld[bld["name"].notna()].copy()
pts = bld.geometry.representative_point()
bld["lat"] = pts.y
bld["lon"] = pts.x
bld["node"] = ox.distance.nearest_nodes(G, X=bld["lon"].values, Y=bld["lat"].values)
bld = bld.reset_index(drop=True)
bld[["name", "lat", "lon", "node"]].sort_values("name").to_csv("data/buildings.csv", index=False)
print("named buildings:", len(bld))

rows = []
for pat in STOP_PATTERNS:
    m = bld[bld["name"].str.contains(pat, case=False, na=False)]
    if m.empty:
        print("NOT FOUND:", pat)
        continue
    if len(m) > 1:
        print("multiple matches for", pat, "->", m["name"].tolist(), "(using first)")
    rows.append(m.iloc[0][["name", "lat", "lon", "node"]].to_dict())

for s in MANUAL_STOPS:
    if s["lat"] is None or s["lon"] is None:
        print("MANUAL STOP has no coordinates, skipped:", s["name"])
        continue
    if any(s["name"] == r["name"] for r in rows):
        continue
    node = ox.distance.nearest_nodes(G, X=s["lon"], Y=s["lat"])
    rows.append({"name": s["name"], "lat": s["lat"], "lon": s["lon"], "node": node})

stops = pd.DataFrame(rows).drop_duplicates(subset="node").reset_index(drop=True)
EXCLUDE = "PENS|PPNS|Airlangga|Mall|Puskesmas|SMP|SMA|Kelurahan|Rumah|Perumdos|Wisma|Balai|Kantor Pos"
dy = (bld["lat"] - CENTER[0]) * 111000
dx = (bld["lon"] - CENTER[1]) * 110000
near = bld[(dx > -480) & (dx < 470) & (dy > -440) & (dy < 520)]
near = near[~near["name"].str.contains(EXCLUDE, case=False, na=False)]
extra = near[["name", "lat", "lon", "node"]]
extra = extra[~extra["node"].isin(stops["node"]) & ~extra["name"].isin(stops["name"])]
extra = extra.drop_duplicates(subset="node").drop_duplicates(subset="name")
extra = extra.sample(n=min(len(extra), 45 - len(stops)), random_state=42)
stops = pd.concat([stops, extra], ignore_index=True)
stops.to_csv("data/stops.csv", index_label="stop_id")
print(stops)


def haversine(u, v):
    lat1, lon1 = math.radians(G.nodes[u]["y"]), math.radians(G.nodes[u]["x"])
    lat2, lon2 = math.radians(G.nodes[v]["y"]), math.radians(G.nodes[v]["x"])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(a))


def astar_length(u, v):
    return nx.astar_path_length(G, u, v, heuristic=haversine, weight="length")


nodes = stops["node"].tolist()
n = len(nodes)
D = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        D[i, j] = D[j, i] = astar_length(nodes[i], nodes[j])

np.save("data/dist_matrix.npy", D)
pd.DataFrame(D, index=stops["name"], columns=stops["name"]).to_csv("data/dist_matrix.csv")
print("distance matrix saved:", D.shape)

fig, ax = ox.plot_graph(G, show=False, close=False, node_size=0, edge_color="#999999", bgcolor="white")
ax.scatter(stops["lon"], stops["lat"], c="red", s=40, zorder=5)
for i, row in stops.iterrows():
    ax.annotate(str(i), (row["lon"], row["lat"]), color="blue", fontsize=9)
fig.savefig("results/campus_stops.png", dpi=200, bbox_inches="tight")
plt.close(fig)
