"""
In this file we define the GNN architecture and all the neural networks that it uses for computations. 
At the bottom there's a small snippet of how to initialize the GNN given a saved graph generated from test.py

To see better how this works I recommend checking out the colab I generated while creating this

GNN:
1. Encoder -> encode the node features and edge features (based on edge type) into the latent space using
  separate syntax

2. Message Passing -> broken up into two steps
  1. (for all edges) Take in edge + nodes -> concat and pass thru NN -> new edge features
  2. (for every node), (for every edge in node.edges) aggregate (sum) each edge type and pass all into NN -> new Node

3. Decoder -> 
want steps 1 and 2 to be separate -> want to aggregate diff edge types separately we do this using a custom
aggregation function defined in bui

TODO:
 - Write skeleton of encoder + message passing (using MLP for everything) -> jax ref code (done)
 - Make sure encoded shape is correct (done)
 - Add layer norm for MLP builders (done)
 - Write a lil preprocessing to convert rod positions -> GT delta velocity (todo)

Later:
 - Accumulated Normalizer of graph features
"""

import jax.numpy as np
import jax
import equinox as eqx
import jraph
from typing import Optional, List



# Want a custom MLP builder that adds layer norm onto MLP final layer for us:
def build_mlp(in_sz, out_sz, width, depth, key, add_layer_norm=True) -> eqx.nn.MLP:
  mlp = eqx.nn.MLP(in_sz, out_sz, width, depth, key=key)

  if add_layer_norm:
    layer_norm = eqx.nn.LayerNorm(out_sz)
    mlp = eqx.nn.Sequential([mlp, layer_norm])
  return mlp


################## Encoder
# Defining the Encoder class:
class Encoder(eqx.Module):
  node_enc_mlp: eqx.nn.MLP
  body_enc_mlp: eqx.nn.MLP
  cable_enc_mlp: eqx.nn.MLP
  con_enc_mlp: eqx.nn.MLP
  _jraph_encoder: jraph.GraphMapFeatures # Store the actual jraph object

  def __call__(self, graph):
    """
    Assumes graph input adheres to the node sz, body sz, etc. inputted when you call create_encoder
    """
    return self._jraph_encoder(graph)
  

# Decided to have this as a non encoder method and let encoder keep it's basic init method
# In case there's a better/different way you want to initialize
def create_encoder(
    latent_dim: int,
    hidden_sz: int,
    mlp_depth: int,
    node_sz: int,
    body_sz: int,
    cable_sz: int,
    con_sz: int,
    key: jax.random.PRNGKey) -> Encoder:
  """
  To create our encoder we first create 4 MLPs:
  One for nodes and each edge type
   - These MLPs will concatenate the array args they recieve by last axis
   - They will encode both nodes as well as edges into the latent dim space
   - For edges we concat onto the last column edge types as jraph doesn't support passing dicts to its aggregate fn
   - We package the encoder and all it's MLPs into a Encoder object

  The output after encoding (assuming that you give a correct graph input) is that
  edges will be of shape (n_edge, latent_dim+1) and nodes will be of shape (n_node, latent_dim)
  """
  keys = jax.random.split(key, 4)
  node_enc_mlp=build_mlp(node_sz, latent_dim, hidden_sz, mlp_depth, key=keys[0])
  body_enc_mlp=build_mlp(body_sz, latent_dim, hidden_sz, mlp_depth, key=keys[1])
  cable_enc_mlp=build_mlp(cable_sz, latent_dim, hidden_sz, mlp_depth, key=keys[2])
  con_enc_mlp=build_mlp(con_sz, latent_dim, hidden_sz, mlp_depth, key=keys[3])

  @jraph.concatenated_args
  def encode_nodes(feats):
    # Need jax.vmap b/c node_mlp just expects input of shape (node_sz,), not (batch_sz, node_sz)
    return jax.vmap(node_enc_mlp)(feats)

  def encode_edges(edges):
    edges =  {
        "body": jax.vmap(jraph.concatenated_args(body_enc_mlp))(edges['body']),
        "cable": jax.vmap(jraph.concatenated_args(cable_enc_mlp))(edges['cable']),
        "con": jax.vmap(jraph.concatenated_args(con_enc_mlp))(edges['con']),
        "edge_type": edges['edge_type']
    }
    edge_matrix = np.concat([edges['body'], edges['cable'], edges['con']], axis=0)
    # concat edge types to last col of edge matrix
    edge_matrix = np.concatenate([edge_matrix, np.reshape(edges['edge_type'], shape=(edges['edge_type'].shape[0], 1))], axis=-1)
    return edge_matrix

  jraph_enc =  jraph.GraphMapFeatures(
      embed_edge_fn = encode_edges,
      embed_node_fn = encode_nodes,
  )
  return Encoder(node_enc_mlp, body_enc_mlp, cable_enc_mlp, con_enc_mlp, jraph_enc)


##################### MESSAGE PASSERS

class MsgPass(eqx.Module):
  edge_update_mlp: eqx.nn.MLP
  node_update_mlp: eqx.nn.MLP
  _jraph_processor: jraph.InteractionNetwork

  def __call__(self, graph):
    return self._jraph_processor(graph)
  

def make_graph_net(
    latent_dim: int,
    hidden_sz: int,
    mlp_depth: int,
    key: jax.random.PRNGKey) -> jraph.GraphNetwork:
  """
  How our graph network works:
  1. We will create an edge update MLP which takes as input a concatenation of edge + sender and reciever node feats
    - We pass the first latent dim cols of edge feats to this MLP, keeping the edge types array the same
  2. We aggregate edges based on edge type
     - achieve this by using a specialized version of segment_sum (more comments and explanation below)
  3. We create a node update MLP which takes nodes feats + aggregated edge feats of all edge types
   - So the final input vector has latent dim * 4 features

  All of this is packaged into a jraph.GraphNetwork object
  """
  keys = jax.random.split(key, 2)

  edge_update_mlp = build_mlp(latent_dim*3, latent_dim, hidden_sz, mlp_depth, key=keys[0])
  node_update_mlp = build_mlp(latent_dim*4, latent_dim, hidden_sz, mlp_depth, key=keys[1])

  def update_edge_fn(
      edge_features,
      sender_node_features,
      receiver_node_features):
    inp = np.concatenate([edge_features[:, :latent_dim], sender_node_features, receiver_node_features], axis=-1)
    edge_features.at[:, :latent_dim].set(jax.vmap(edge_update_mlp)(inp))
    return edge_features

  def aggregate_edges_fn(
    edge_data,
    node_idx,
    n_nodes):
    # Use a specialized version of segment_sum in order to aggregate different edge types
    edge_types = edge_data[:, -1]
    feats = edge_data[:, :latent_dim]

    # Instead of just using n_nodes as number of segments we want
    # we say that for each node there's 3 segments we want, i.e. 1 for each edge type
    num_segments = n_nodes * 3

    # The segment ids are then the node idx scaled by 3 + the edge type
    segment_ids = node_idx * 3 + edge_types 
    return jraph.segment_sum(feats, segment_ids.astype(int), num_segments) # gives shape (num_segments, latent dim)

  def update_node_fn(
      node_features,
      agg_reciever_feats):
    # To update our nodes we only aggregate based on that nodes' reciever edges
    # We reshape this s.t. there's num_edge_type segments per node (so 3 = num edge types in this implementation)
    agg_edges = np.reshape(agg_reciever_feats, shape=(node_features.shape[0],3, -1))

    # Add an extra dimension into node features so that we can concat them with their aggregated edge feats
    node_feats = np.reshape(node_features, shape=(node_features.shape[0], 1, -1))
    inp = np.concatenate([node_feats, agg_edges], axis=1)

    # Lastly we flatten the input to get shape (n_nodes, concatted node and agg edge feats) and send this into the MLP
    inp = np.reshape(inp, shape=(node_feats.shape[0], -1))
    return jax.vmap(node_update_mlp)(inp)

  net = jraph.InteractionNetwork(
      update_edge_fn = update_edge_fn,
      update_node_fn = update_node_fn,
      aggregate_edges_for_nodes_fn = aggregate_edges_fn,
      include_sent_messages_in_node_update=False # only use reciever edges for node updates
  )

  return MsgPass(edge_update_mlp, node_update_mlp, net)


##################### FULL GNN
class CustomGNN(eqx.Module):
  encoder: Encoder
  processor: list[MsgPass]
  decoder_mlp: eqx.nn.MLP
  decoder: jraph.GraphMapFeatures

  def __init__(self,
               latent_dim,
               hidden_dim,
               mlp_depth,
               n_mp_steps,
               node_sz, body_sz, cable_sz, con_sz, key=jax.random.key(0)):
    keys = jax.random.split(key, 3)
    self.encoder = create_encoder(latent_dim, hidden_dim, mlp_depth,
                                  node_sz, body_sz, cable_sz, con_sz, keys[0])

    self.processor = []
    for i in range(n_mp_steps):
      self.processor.append(make_graph_net(latent_dim, hidden_dim, mlp_depth, keys[1]))

    self.decoder_mlp = build_mlp(latent_dim, 3, hidden_dim, mlp_depth, key=keys[2], add_layer_norm=False)
    self.decoder = jraph.GraphMapFeatures(
        embed_node_fn = jax.vmap(self.decoder_mlp),
    )


  def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    latent_graph = self.encoder(graph)

    inp_graph = latent_graph

    for mp_step in self.processor:

      proc_graph = mp_step(inp_graph)

      # Add residual connections
      proc_graph._replace(nodes = proc_graph.nodes + inp_graph.nodes)
      proc_graph._replace(edges = proc_graph.edges + inp_graph.edges)

      inp_graph = proc_graph

    node_dv_preds = self.decoder(proc_graph).nodes
    return node_dv_preds


def get_sz(graph) -> list:
    """
    Get node, body, cable, and contact concatenated input size
    for inputting to the GNN encoder
    """
    node_sz = sum(n.shape[-1] for n in graph.nodes.values())
    body_sz = sum(n.shape[-1] for n in graph.edges['body'].values())
    cable_sz = sum(n.shape[-1] for n in graph.edges['cable'].values())
    con_sz = sum(n.shape[-1] for n in graph.edges['con'].values())

    return [node_sz, body_sz, cable_sz, con_sz]




if __name__ == '__main__':
  from save import load_graph_tuple

  graph = load_graph_tuple('tst_graph_2.bin')
  sizes = get_sz(graph)
  latent_dim, hidden_dim, mlp_depth, n_mp_steps = 64, 128, 1, 2
  key = jax.random.key(1)
  gnn = CustomGNN(latent_dim, hidden_dim, mlp_depth, n_mp_steps, *sizes, key)




