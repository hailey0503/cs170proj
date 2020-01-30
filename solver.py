import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils

from student_utils import *
from sklearn.cluster import KMeans
import math
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min

"""
======================================================================
  Complete the following function.
======================================================================
"""

def check_valid_walk(G, closed_walk):
    if len(closed_walk) == 2:
        print(closed_walk[0] == closed_walk[1])
    print([(closed_walk[i], closed_walk[i+1]) in G.edges for i in range(len(closed_walk) - 1)])


# source: https://www.learndatasci.com/tutorials/k-means-clustering-algorithms-python-intro/
def graph_to_edge_matrix(G):
    """Convert a networkx graph into an edge matrix.
    See https://www.wikiwand.com/en/Incidence_matrix for a good explanation on edge matrices
   
    Parameters
    ----------
    G : networkx graph
    """
    # Initialize edge matrix with zeros
    edge_mat = np.zeros((len(G), len(G)), dtype=int)

    # Loop to set 0 or 1 (diagonal elements are set to 1)
    for node in G:
        for neighbor in G.neighbors(node):
            edge_mat[node][neighbor] = 1
        edge_mat[node][node] = 1

    return edge_mat

def get_centeroids(edge_matrix):
    k = int(math.sqrt(len(edge_matrix[0])/2))
    # print(f'k: {k}')
    kmeans = KMeans(n_clusters=k).fit(edge_matrix)
    # print(kmeans.labels_)
    cluster_dict = {}
    for i in range(len(kmeans.labels_)):
        if kmeans.labels_[i] not in cluster_dict:
            cluster_dict[kmeans.labels_[i]] = []     
        cluster_dict[kmeans.labels_[i]].append(i)
    # print(cluster_dict)

    #source https://codedump.io/share/XiME3OAGY5Tm/1/get-nearest-point-to-centroid-scikit-learn
    centeroids, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, edge_matrix)
    centeroids = centeroids.tolist()
    # print(centeroids)

    return centeroids, kmeans.labels_, cluster_dict

def get_homes_from_cluster(cluster_dict, cluster_idx, homes_indices):
    return list(set(cluster_dict[cluster_idx]) & set(homes_indices)) 

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """

    #convert adjacency_matrix into graph
    G, message = adjacency_matrix_to_graph(adjacency_matrix)
    # print(f'graph: {G.nodes()}')
    # print(f'message: {message}')
    # print(f'edges: {G.edges}')

    # print(f'is metric: {is_metric(G)}')

    edge_mat = graph_to_edge_matrix(G)
    # print(edge_mat)
    centeroids, labels, cluster_dict = get_centeroids(edge_mat)

    node_index_to_name_dict = {}
    node_name_to_index_dict = {}
    for i in range(len(list_of_locations)):
        node_index_to_name_dict[i] = list_of_locations[i]
        node_name_to_index_dict[list_of_locations[i]] = i

    homes_indices = convert_locations_to_indices(list_of_homes, list_of_locations)
    
    source = node_name_to_index_dict[starting_car_location]
    # print(f'source: {source}')
    path = [node_name_to_index_dict[starting_car_location]]

    predecessors, shortest = nx.floyd_warshall_predecessor_and_distance(G)
    shortest = dict(shortest)
    drop_offs = {}


    while len(centeroids) > 0:
        # print(f'centeroids: {centeroids}')
        distance = float('inf')
        next = None
        for target in centeroids:
            if shortest[source][target] < distance:
                next = target
                distance = shortest[source][target]

        # print(f'source: {source} next: {next}')
        path_segment = nx.reconstruct_path(source, next, predecessors)
        # print(path_segment)
        path += path_segment[1:]
        
        homes = get_homes_from_cluster(cluster_dict, labels[next], homes_indices)
        drop_offs[next] = homes

        centeroids.remove(next)
        source = next

    path_segment = nx.reconstruct_path(source, path[0], predecessors)
    path += path_segment[1:]
    # print(path)

    # check_valid_walk(G, path)
    # print(drop_offs)

    # nx.draw_networkx(G)
    # plt.show()

    return (path, drop_offs)

"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
