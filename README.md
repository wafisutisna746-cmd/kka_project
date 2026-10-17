# KKA Midterm project

|    NRP     |           Nama             |
| :--------: |       :------------:       |
| 5025251002 | Muhammad Faris Alfarrel           |
| 5025251011 | Dzulfiqar Rafi`ussunnah   |
| 5025251158 | Aditya Lingga Mardika                   |
| 5025251017 | Wafi Fawwaz Sutisna                    |


# ITS Campus Multi-Stop Route Optimization

Midterm project for the Artificial Intelligence course (myITS Classroom). We study the problem of planning the shortest walking route that visits several buildings on the ITS Surabaya (Sukolilo) campus and returns to the start, and compare four search and optimization methods on it.

## Problem

Students, staff, and visitors often need to visit several places in one trip (for example a faculty, the library, the rectorate, and the mosque). Choosing the order of visits by hand is inefficient, and the number of possible orders grows factorially with the number of stops. We model this as a Traveling Salesman Problem (TSP) on the real campus walking network:

- Pairwise walking distances between buildings are computed with A* on an OpenStreetMap graph.
- The visiting order is optimized with four methods, which are compared on solution quality, consistency, and runtime as the number of stops grows.

## Data

All data comes from OpenStreetMap through [OSMnx](https://github.com/gboeing/osmnx) and is downloaded when `setup_campus_graph.py` runs.

- Walking network within a 1.5 km radius of the campus center (-7.2819, 112.7948), converted to an undirected graph, largest connected component only: 1892 nodes, 2494 edges. Edge weight is length in meters.
- Named OSM buildings, each snapped to its nearest graph node. 15 stops are chosen by name, and the rest are added from named buildings inside a bounding box around the campus (off-campus names are excluded), for 45 stops in total.
- For each experiment size n (10, 20, 30, 40), a random subset of n stops is drawn from the 45 stops with a fixed seed (0).

OpenStreetMap is edited continuously, so re-downloading later may give slightly different building names and counts.

## Methods

**Distance matrix.** A* with a Haversine (great-circle) heuristic computes the shortest walking distance between every pair of stops. This is admissible because the straight-line distance never exceeds the path length along the network.

**TSP formulation.** A tour is a permutation of the stops, treated as a closed cycle that returns to the first stop. The cost is the total walking distance in meters.

| Method | Description |
|---|---|
| Nearest Neighbor | Greedy baseline, always moves to the closest unvisited stop (start: stop 0). Deterministic. |
| Simulated Annealing (SA) | 2-opt (segment reversal) neighbor, geometric cooling from T0 = 5% of the initial tour cost down to 0.01, 20,000 iterations. |
| Genetic Algorithm (GA) | Population 100, 200 generations, tournament selection (k = 3), order crossover (rate 0.9), swap or reversal mutation (rate 0.2), elitism of 2. |
| GA + 2-opt (modified) | Memetic variant of the GA: 2-opt local search is applied to every initial individual and every child. Population 30, 100 generations. |

SA and GA use the same budget of 20,000 tour evaluations. The GA + 2-opt variant uses a smaller population because each local search is expensive, so its cost is better compared by runtime than by evaluation count.

**Reference solution.** For up to 16 stops, the exact optimum is computed with the Held-Karp dynamic program. For larger sizes this is too slow, so the reference is the best tour found by any method over all runs (the best known solution).

## Experiments

- Sizes: 10, 20, 30, 40 stops.
- 10 runs per stochastic method (seeds 0 to 9). Nearest Neighbor runs once.
- Metrics: tour length, optimality gap against the reference, runtime.

## Results

Mean optimality gap (lower is better):

| Stops | Nearest Neighbor | GA | SA | GA + 2-opt | Best known tour (m) |
|---|---|---|---|---|---|
| 10 | 8.78% | 0.00% | 0.00% | 0.00% | 5157.7 |
| 20 | 6.93% | 0.39% | 0.00% | 0.00% | 7699.1 |
| 30 | 14.89% | 0.31% | 0.07% | 0.00% | 8749.4 |
| 40 | 18.53% | 2.11% | 0.22% | 0.00% | 10016.9 |

Mean runtime per run at 40 stops: Nearest Neighbor about 0 s, SA 0.31 s, GA 0.49 s, GA + 2-opt 3.07 s.

[Convergence at 40 stops]

<img width="1428" height="942" alt="convergence" src="https://github.com/user-attachments/assets/aa132522-4930-4083-95f1-73ae449cd14c" />


Observations:

- Nearest Neighbor gets worse as the problem grows, because early greedy choices leave distant stops for the end.
- GA + 2-opt found the best known tour in every run at every size, but takes roughly ten times longer than SA.
- SA gives the best quality for its cost: within 0.22% of the best known tour at 40 stops in about 0.3 s.
- The plain GA is the weakest of the three metaheuristics, with a larger gap and more variance at 40 stops, since crossover and mutation alone do not fine-tune a tour as well as local search.

Limitations:

- Above 16 stops the reference is the best tour found by the methods themselves, not a proven optimum, so the 0% gap of GA + 2-opt means it matched the best known solution.
- In the convergence plot, the x-axis counts tour evaluations made by the GA loop only and does not include evaluations inside 2-opt, so the GA + 2-opt curve should not be read as better sample efficiency than SA.
- Distances are walking distances on the OSM network. Travel time, congestion, covered walkways, and class schedules are not modeled.
- Building positions are snapped to the nearest graph node, so very close buildings can share a node.

## Repository structure

```
its-campus-route/
  setup_campus_graph.py   download graph, select stops, build distance matrix
  run_experiments.py      run all methods and produce results
  requirements.txt
  data/
    its_walk.graphml      campus walking graph
    buildings.csv         all named buildings found, with coordinates and graph node
    stops.csv             selected stops
    dist_matrix.npy       pairwise A* distance matrix (meters)
    dist_matrix.csv       same matrix with building names
  results/
    results.csv           every run: size, method, seed, cost, gap, time, tour
    summary.csv           mean, std, best, gap, and time per size and method
    convergence.png       convergence curves at the largest size
    campus_stops.png      map of the selected stops
```

## How to run

Python 3.10 or newer is recommended. Run the commands from inside this folder, because file paths are relative.

```
pip install -r requirements.txt
py setup_campus_graph.py
py run_experiments.py
```

On macOS or Linux, use `python3` instead of `py`.

`setup_campus_graph.py` needs an internet connection and takes a minute or two. To change the stops, edit `STOP_PATTERNS` (names matched against `data/buildings.csv`) and the `EXCLUDE` list or the bounding box in that script. To change the experiment sizes or budgets, edit the constants at the top of `run_experiments.py`.

## Tools

Python, OSMnx, NetworkX, NumPy, pandas, Matplotlib, scikit-learn.
