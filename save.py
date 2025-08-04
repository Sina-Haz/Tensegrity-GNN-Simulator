"""
Implement functions to save and reload graphs using jax utilities
"""


import jraph
from flax.serialization import to_bytes, from_bytes
import jax.numpy as np
import jax

def save_graph_tuple(graph: jraph.GraphsTuple, filename: str) -> None:
    """Save a jraph.GraphsTuple to a binary file."""
    with open(filename, "wb") as f:
        f.write(to_bytes(graph))
    print(f"Graph saved to {filename}")

def load_graph_tuple(filename: str) -> jraph.GraphsTuple:
    """Load a jraph.GraphsTuple from a binary file."""
    with open(filename, "rb") as f:
        graph = from_bytes(jraph.GraphsTuple, f.read())
    print(f"Graph loaded from {filename}")
    graph = jraph.GraphsTuple(**graph)
    return graph

def graphs_equal(g1: jraph.GraphsTuple, g2: jraph.GraphsTuple) -> bool:
    # Flatten the graphs into pytrees
    flat1, _ = jax.tree_util.tree_flatten(g1)
    flat2, _ = jax.tree_util.tree_flatten(g2)

    if len(flat1) != len(flat2):
        return False

    for a, b in zip(flat1, flat2):
        # compare arrays with tolerance, fallback to equality
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if not np.array_equal(a, b):
                return False
        else:
            # Can cause problems
            if a != b:
                return False
    return True