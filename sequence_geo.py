import copy
import itertools
import logging
from pathlib import Path
import time

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

# set up basic logging
logging.basicConfig(level=logging.INFO)


# define a record reader to avoid loading all of a layers attributes. This is purely to save memory, and won't 
# result in any speedup, but helps manage with large datasets.
# Credit to https://gis.stackexchange.com/questions/129414/only-read-specific-attribute-columns-of-a-shapefile-with-geopandas-fiona
def records(filename, usecols, **kwargs):
  """Read geometry records from a data source, gathering only the desired columns.

  Parameters:
    filename: the filename to read data from
    usecols: the columns to pull out of the data source
    kwargs: any additional arguments that should be passed through to fiona
  Returns:
    feature: a geojson feature with the usecols as properties
  """

  logging.debug("Reading geometry from %s", filename)
  with fiona.open(filename, **kwargs) as source:
    for feature in source:
      f = {k: feature[k] for k in ['id', 'geometry']}
      f['properties'] = {k: feature['properties'][k] for k in usecols}
      yield f


def table(filename, usecols, **kwargs):
  """Read a table from a data source, gathering only the desired columns.

  Parameters:
    filename: the filename to read data from
    usecols: the columns to pull out of the data source
    kwargs: any additional arguments that should be passed through to fiona
  Returns:
    feature: a geojson feature with the usecols as properties
  """

  logging.debug("Reading table from %s", filename)
  with fiona.open(filename, **kwargs) as source:
    for feature in source:
      f = {k: feature['properties'][k] for k in usecols}
      yield f


def get_all_block_faces(path, ngdal='ROADS', bf='BF', lb='LB', lu='LU'):
  """Read all the source data and generate a list of all the block faces to be sequenced.

  Parameters:
    path: the path to the source database
    ngdal: NGD_AL layer name
    bf: block face (BF) layer name
    lb: listing block (LB) layer name
    lu: listing unit (LU) layer name
  Returns:
    lines: GeoDataFrame representing every line in the source data
  """

  logging.debug("get_all_block_faces start")

  # the column to use for calculating weights
  weight_field_name = 'SHAPE_Length'

  # load the road network
  logging.debug("Loading NGD_AL data")
  road_columns = ['NGD_UID',weight_field_name]
  lines = gpd.GeoDataFrame.from_features(records(path, road_columns, layer=ngdal))

  # load the block face table
  logging.debug("Loading BF data")
  bf_columns = ['NGD_UID', 'BF_UID', 'BB_UID', 'ARC_SIDE', 'LB_UID']
  bf_all = pd.DataFrame.from_records(table(path, bf_columns, layer=bf))
  bf_all['ARC_SIDE'] = bf_all['ARC_SIDE'].astype('category')

  # load the listing block data
  logging.debug("Loading LB data")
  lb_columns = ['LB_UID', 'LU_UID','S_FLAG','LFS_UID']
  lb_all = gpd.GeoDataFrame.from_features(records(path, lb_columns, layer=lb))
  # drop the geometry - it won't be used
  lb_all.drop(columns='geometry', inplace=True)
  lb_all['S_FLAG'] = lb_all['S_FLAG'].astype('category')

  # merge everything onto the line network
  # NOTE: This is only going to end up with lines that are block faces. For BOs as well, use "how=left"
  lines = lines.merge(bf_all, on='NGD_UID', sort=False)
  lines = lines.merge(lb_all, on='LB_UID', sort=False, copy=False)

  logging.debug("get_all_block_faces end")
  return lines


def get_shortest_paths_distances(graph, pairs, edge_weight_name):
  """Compute the shortest distance between each pair of nodes in a graph.
  
  Returns a dictionary keyed on node pairs (tuples).
  """

  logging.debug("get_shortest_paths_distances start")
  distances = {}
  for pair in pairs:
    distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)

  logging.debug("get_shortest_paths_distances end")
  return distances
  

def create_complete_graph(pair_weights, flip_weights=True):
  logging.debug("create_complete_graph start")
  g = nx.Graph()
  for k, v in pair_weights.items():
    wt_i = - v if flip_weights else v
    g.add_edge(k[0], k[1], **{'distance': v, 'weight': wt_i})
  
  logging.debug("create_complete_graph end")
  return g


def add_augmenting_path_to_graph(graph, min_weight_pairs):
  """Add the min weight matching edges to the original graph.
  Parameters:
    graph: NetworkX graph
    min_weight_pairs: list[tuples] of node pairs from min weight matching
  Returns:
    augmented NetworkX graph
  """
  
  logging.debug("add_augmenting_path_to_graph start")
  # use a MultiGraph to allow for parallel edges
  graph_aug = nx.MultiGraph(graph.copy())
  for pair in min_weight_pairs:
    graph_aug.add_edge(pair[0],
                      pair[1],
                      **{'distance': nx.dijkstra_path_length(graph, pair[0], pair[1]),
                        'trail': 'augmented'}
                      )
  
  logging.debug("add_augmenting_path_to_graph end")
  return graph_aug


def create_eulerian_circuit(graph_augmented, graph_original, weight_field_name, start_node=None):
  """Create the eulerian path using only edges from the original graph."""

  logging.debug("create_eulerian_circuit start")
  euler_circuit = []
  naive_circuit = list(nx.eulerian_circuit(graph_augmented, source=start_node))
  
  for edge in naive_circuit:
    # get the original edge data
    edge_data = graph_augmented.get_edge_data(edge[0], edge[1])
    
    # this is not an augmented path, just append it to the circuit
    if edge_data[0].get('trail') != 'augmented':
      edge_att = graph_original[edge[0]][edge[1]]
      # appends a tuple to the final circuit
      euler_circuit.append((edge[0], edge[1], edge_att))
      continue
  
    # edge is augmented, find the shortest 'real' route
    aug_path = nx.shortest_path(graph_original, edge[0], edge[1], weight=weight_field_name)
    aug_path_pairs = list(zip(aug_path[:-1], aug_path[1:]))

    logging.debug('Filling in edges for augmented edge: %s', edge)

    # add the edges from the shortest path
    for edge_aug in aug_path_pairs:
        edge_aug_att = graph_original[edge_aug[0]][edge_aug[1]]
        euler_circuit.append((edge_aug[0], edge_aug[1], edge_aug_att))
  
  logging.debug("create_eulerian_circuit end")
  return euler_circuit


def create_cpp_edgelist(euler_circuit):
  """
  Create the edgelist without parallel edge for the visualization
  Combine duplicate edges and keep track of their sequence and # of walks
  Parameters:
      euler_circuit: list[tuple] from create_eulerian_circuit
  """

  logging.debug("create_cpp_edgelist start")
  cpp_edgelist = {}

  for i, e in enumerate(euler_circuit):
    edge   = frozenset([e[0], e[1]])
    
    # each edge can have multiple paths (L/R), so number accordingly
    if edge not in cpp_edgelist:
      cpp_edgelist[edge] = e
      # label the right edge with the sequence number
      # this implements the 'right hand rule'
      for j, bf in enumerate(cpp_edgelist[edge][2]):
        if cpp_edgelist[edge][2][j]['ARC_SIDE'] == 'R':
          cpp_edgelist[edge][2][j]['sequence'] = i
          cpp_edgelist[edge][2][j]['visits'] = 1
    else:
      # label the other edge with a sequence number
      for j, bf in enumerate(cpp_edgelist[edge][2]):
        if not cpp_edgelist[edge][2][j].get('sequence'):
          cpp_edgelist[edge][2][j]['sequence'] = i
          cpp_edgelist[edge][2][j]['visits'] = 1
          continue

  logging.debug("create_cpp_edgelist end")
  return list(cpp_edgelist.values())


def flatten_edgelist(edgelist):
  """Turn a MultiGraph edge list into a flattened list."""

  logging.debug("flatten_edgelist start")
  for multiedge in edgelist:
    source = multiedge[0]
    target = multiedge[1]
    for edge in multiedge[2]:
      edge_attribs = multiedge[2][edge]
      edge_attribs['source_x'] = source[0]
      edge_attribs['source_y'] = source[1]
      edge_attribs['target_x'] = target[0]
      edge_attribs['target_y'] = target[1]
      yield edge_attribs
  
  logging.debug("flatten_edgelist end")


def sequence_edges(g, weight_field_name):
  """Develop an ordered edge list that represents the sequence through an LU."""
  
  logging.debug("sequence_edges start")

  # Step 1: find nodes of odd degree
  logging.debug("Finding nodes of odd degree")
  nodes_odd_degree = [v for v,d in g.degree() if d % 2 == 1]

  # Step 2: find minimum distance pairs
  # 2.1: Compute all node pairs
  logging.debug("Finding odd node pairs")
  odd_node_pairs = list(itertools.combinations(nodes_odd_degree, 2))
  if len(odd_node_pairs) > 100:
    logging.warning("%s node pairs will take a while to process", len(odd_node_pairs))
  
  # 2.2: Compute shortest paths between node pairs
  logging.debug("Computing shortest paths between node pairs")
  odd_node_pairs_shortest_paths = get_shortest_paths_distances(g, odd_node_pairs, weight_field_name)

  # 2.3: Create the complete graph
  logging.debug("Creating a complete graph from shortest paths")
  g_odd_complete = create_complete_graph(odd_node_pairs_shortest_paths)

  # 2.4: Compute minimum weight matching
  logging.debug("Computing minimum weight matches")
  odd_matching_dupes = nx.algorithms.max_weight_matching(g_odd_complete, True)
  # Convert the dictionary to a list of tuples and dedupe since this is undirected
  odd_matching = list(pd.unique([tuple(sorted([k,v])) for k, v in odd_matching_dupes]))

  # 2.5: Augment the original graph
  logging.debug("Augmenting original graph with minimum weight matches")
  g_aug = add_augmenting_path_to_graph(g, odd_matching)

  # Step 3: Compute Eulerian circuit
  logging.debug("Finding start node in NE corner")
  start_point = max(g_aug.nodes())

  logging.debug("Computing naive Eulerian circuit")
  naive_euler_circuit = list(nx.eulerian_circuit(g_aug, source=start_point))

  logging.debug("Computing eulerian circuit")
  euler_circuit = create_eulerian_circuit(g_aug, g, weight_field_name, start_point)

  logging.debug("Get edge list for circuit")
  cpp_edgelist = create_cpp_edgelist(euler_circuit)

  logging.debug("Flattening edge list into final sequence")
  edge_sequence = pd.DataFrame.from_records(flatten_edgelist(cpp_edgelist))
  edge_sequence.sort_values(by='sequence', inplace=True)

  logging.debug("sequence_edges end")
  return edge_sequence


def calculate_colcodes(edge_sequence):
  """Calculate the collection order for blocks in a given edge sequence."""

  logging.debug("calculate_colcodes start")

  lb_grouped = edge_sequence.groupby('LB_UID', sort=False)
  lb_colcode = 1
  for name, group in lb_grouped:
    edge_sequence.loc[edge_sequence['LB_UID'] == name, 'lb_colcode'] = lb_colcode
    edge_sequence.loc[edge_sequence['LB_UID'] == name, 'bf_colcode'] = range(1, len(group)+1)
    lb_colcode += 1
  
  # final edge sequence
  edge_sequence['chain_id'] = np.where(edge_sequence['bf_colcode'] == 1, 1, 0)
  edge_sequence['bf_colcode'] = edge_sequence['bf_colcode'].astype('int')
  edge_sequence['lb_colcode'] = edge_sequence['lb_colcode'].astype('int')

  logging.debug("calculate_colcodes end")
  return edge_sequence

def main():
  """Actual start of the process."""
  # data paths
  in_data_path = r'testdata/Sample_data.gdb'
  bf_layer_name = 'BF'
  ngdal_layer_name = 'ROADS'
  lb_layer_name = 'LB'
  lu_layer_name = 'LU'
  # output file
  out_path = 'final_sequence.csv'
  # the column to use for calculating weights
  weight_field_name = 'SHAPE_Length'

  # get all the roads to route through (entire dataset)
  logging.info("Finding all roads in data set")
  lines = get_all_block_faces(in_data_path, ngdal_layer_name, bf_layer_name, lb_layer_name, lu_layer_name)

  # need to specifically pull the start and end nodes. networkx doesn't know how to load a GeoDataFrame
  logging.info("Calculating start points for road segments")
  lines['start_node'] = lines.geometry.apply(lambda x: x[0].coords[0])
  lines['end_node'] = lines.geometry.apply(lambda x: x[0].coords[-1])
  # generate a unique identifier for each node (start and end have unique IDs)
  id_scope = len(lines)*2
  lines['start_node_id'] = range(1, id_scope, 2)
  lines['end_node_id'] = range(2, id_scope+1, 2)

  # things are actually sequenced by LU, so group the records and route each one
  logging.info("Grouping roads by LU_UID")
  lu_grouped = lines.groupby('LU_UID')

  # a place to store the full sequencing output from all LUs
  full_sequence = pd.DataFrame()

  logging.info("Processing each LU")
  for lu_uid, lu_group in lu_grouped:
    # if lu_uid != 54118:
    #   continue
    logging.info("Processing LU %s", lu_uid)
    # generate a MultiGraph from the road information in this LU
    g = nx.convert_matrix.from_pandas_edgelist(lu_group, 'start_node', 'end_node', True, nx.MultiGraph)

    # ensure this graph is connected, otherwise tell the user and move on
    if not nx.is_connected(g):
      for c in nx.connected_components(g):
        g_sub = g.subgraph(c)
        edge_sequence = sequence_edges(g_sub, weight_field_name)
        edge_sequence = calculate_colcodes(edge_sequence)
        full_sequence = pd.concat([full_sequence, edge_sequence], ignore_index=True, sort=False)
    else:
      # get an edge sequence for the graph
      edge_sequence = sequence_edges(g, weight_field_name)

      # calculate COLCODE fields
      edge_sequence = calculate_colcodes(edge_sequence)

      # append to the final sequencing output
      logging.debug("Saving sequence to final outputs")
      full_sequence = pd.concat([full_sequence, edge_sequence], ignore_index=True, sort=False)
  
  # write out the results
  logging.info("Writing output to %s", out_path)
  desired_columns = ['LU_UID','LB_UID','lb_colcode','BF_UID','bf_colcode','chain_id','start_node_id','source_x','source_y','end_node_id', 'target_x', 'target_y', 'ARC_SIDE']
  full_sequence.to_csv(out_path, columns=desired_columns, index=False)


if __name__ == '__main__':
  start_time = time.time()

  main()

  end_time = time.time()
  run_seconds = end_time - start_time
  run_time = time.gmtime(run_seconds)
  logging.info("Processing time: {}".format(time.strftime("%H:%M:%S", run_time)))

# Notes
# what happens when a child geography has no roads?
# what happens when a parent geography has no roads?
# what happens when a parent geography has no children? Is this possible?