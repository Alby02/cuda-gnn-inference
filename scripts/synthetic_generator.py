import argparse #read from command line
import os #output directory creation
import struct
import numpy as np
import scipy.sparse as sp #for CSC
import networkit as nk #for graph generation
from export_graph_csc import export_graph_csc #for I/O
from export_dense_matrix import export_dense_matrix




def graph_to_csc(g: "nk.Graph", num_nodes: int, is_directed: bool) -> sp.csc_matrix:
     ## convertion from Networkit Graph to CSC matrix format

     ## alternative much more efficient than nk.algebraic.adjacencyMatrix(): it iterates over the
     # edges (g.iterEdges()), print out (u, v) and it leaves to scipy the job to build the CSC, reducing overhead


    num_edges = g.numberOfEdges()

    # Allocation of source and destination array
    s = np.empty(num_edges, dtype=np.int64)
    d = np.empty(num_edges, dtype=np.int64)


    # Exctracting the edges to put them in source and destination arrays
    i = 0
    for u, v in g.iterEdges(): # edge extraction using g.iterEdges() --> NetworKit iteration

        s[i] = u
        d[i] = v
        i += 1

    if not is_directed: #adding the opposite direction in order for getInNeighbors() to work properly
        s, d = np.concatenate([s, d]), np.concatenate([d, s])


    weights = np.ones(len(s), dtype=np.float32) #weights = 1, to be coherent with ogb and planetoid datasets

    return sp.csc_matrix((weights, (s, d)), shape=(num_nodes, num_nodes))


def generate_synthetic_graph(
    graph_type: str,
    num_nodes: int,
    feature_dim: int,
    out_prefix: str,
    is_directed: bool = False,
    seed: int = 42,
    **kwargs
):

    ## generation of graph topology and feature nodes matrix and export to dense matrix and graph csc format


    nk.setSeed(seed, False) #setting the seed for reproducibility


    rng = np.random.default_rng(seed) #features NumPy generator

    effective_directed = is_directed #flag to handle directed/undirected graph

    if graph_type == "erdos_renyi": # ERDOS RENYI GENERATOR
        p = kwargs.get("p", 0.001) #getting the probability param
        G = nk.generators.ErdosRenyiGenerator(num_nodes, p, directed=is_directed, selfLoops=False).generate()

    elif graph_type == "barabasi_albert": #BARABASI ALBERT GENERATOR
        m = kwargs.get("m", 5)
        effective_directed = False #undirected
        G = nk.generators.BarabasiAlbertGenerator(k=m, nMax=num_nodes, n0=0).generate()
        # params: k = number of edges created towards existing nodes, nMax = total number of nodes to reach,
        # n0 = 0 --> automatic choice of initial graph dimensions


    elif graph_type == "watts_strogatz": #WATTS STROGATZ GENERATOR
        k = kwargs.get("k", 6)
        p = kwargs.get("p", 0.1)
        effective_directed = False #intrinsecally undirected
        G = nk.generators.WattsStrogatzGenerator(nNodes=num_nodes, nNeighbors=k // 2, p=p).generate()
        #params: k = total degree of each node in the initial setup, nNeighbors = only a part of the initial setup
        #

    else:
        raise ValueError(f"ERROR: {graph_type} is not supported")


    #EXPORT OF THE GRAPH IN CSC FORMAT

    csc_mat = graph_to_csc(G, num_nodes, effective_directed) #convertion to CSC

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    graph_filename = f"{out_prefix}.bin_graph"
    export_graph_csc( #export
        filename=graph_filename,
        num_nodes=csc_mat.shape[0],
        col_ptr=csc_mat.indptr,
        row_ind=csc_mat.indices,
        is_directed=effective_directed
    )

    #EXPORT OF DENSE FEATURE MATRIX
    #node features matrix generation
    node_features = rng.standard_normal((num_nodes, feature_dim)).astype(np.float32)

    feats_filename = f"{out_prefix}_feats.bin_matrix"
    export_dense_matrix(feats_filename, node_features) #export


if __name__ == "__main__": #launch from command line
    parser = argparse.ArgumentParser(description="Synthetic Graph Generator")
    parser.add_argument("--type", choices=["barabasi_albert", "erdos_renyi", "watts_strogatz"],
                        default="barabasi_albert", help="Graph type")
    parser.add_argument("--nodes", type=int, default=10000, help="Total nodes")
    parser.add_argument("--feature_dim", type=int, default=128, help="Dimension of the features for each node")
    parser.add_argument("--out_prefix", type=str, default="./synth_data/graph", help="Output path prefix")
    parser.add_argument("--directed", action="store_true", help="Creates an oriented graph")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")

    parser.add_argument("--m", type=int, default=5, help="Parameter m for Barabási-Albert")
    parser.add_argument("--p", type=float, default=0.001, help="Probability parameter p for Erdős-Rényi/Watts-Strogatz")
    parser.add_argument("--k", type=int, default=6, help="Paramter k for Watts-Strogatz")

    args = parser.parse_args()

    generate_synthetic_graph(
        graph_type=args.type,
        num_nodes=args.nodes,
        feature_dim=args.feature_dim,
        out_prefix=args.out_prefix,
        is_directed=args.directed,
        seed=args.seed,
        m=args.m,
        p=args.p,
        k=args.k
    )