import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt

from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""
# function to assign centriod for each cluster
def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

# update the centriods for better approximation
def update(k, centrioddict):
    for i in centrioddict.keys():
        centrioddict[i][0] = np.mean(df[df['closest'] == i]['x'])
        centrioddict[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

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
        A list of (location, [homes]) representing drop-offs
    """
    dict_locations={}
    locations = list(list_of_locations)
    homes = list(list_of_homes)
    start = starting_car_location
    para=params

    #convert adjacency_matrix into graph
    G = adjacency_matrix_to_graph(adjacency_matrix)

    #Checking to see if the graph is metric
    boo = is_metric(G)

    if boo:
        # create a mapping of integers to the names of the location
        for i in range(len(locations)):
            dict_locations[i] = locations[i]
        # convert the locations into coordinates
        # ??

        # Initialize data points
        # change later
        df = pd.DataFrame({
            'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
            'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
        })

        k = len(homes)

        centroids = {
            i + 1: [np.random.randint(0, len(locations)), np.random.randint(0, len(locations))]
            for i in range(k)
        }

        # update new centriod
        centroids = update(centroids)

        # Repeat Assigment Stage many times
        df = assignment(df, centroids)

        # continue with more updates until centriod
        while True:
            closest_centroids = df['closest'].copy(deep=True)
            centroids = update(centroids)
            df = assignment(df, centroids)
            if closest_centroids.equals(df['closest']):
                break
    # centriods will be a dictionary of float value, we need to match to closest location
        # create a rounding function -> float to integer

    # closest_centriods will return a dictionary of keys = location and value is its closest centriod

    # create a MST starting from the source node using dijkstra and heurstics, aka edgeWeights

    # make a cycle by finding a backedge with most minimum weight to source

    # run cost algo on this cycle -> returning a dictionary of dropoffs

    # 1) return the tour
    # 2) return the dropoff locations, returned after running the cost algo



    #pass
    

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
    output_filename = utils.input_to_output(filename)
    output_file = f'{output_directory}/{output_filename}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
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