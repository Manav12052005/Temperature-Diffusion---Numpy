## COMP1730/6730 Project assignment

# Your ANU ID: u7782612
# Your NAME: Manav Singh
# I declare that this submission is my own work
# [ref: https://www.anu.edu.au/students/academic-skills/academic-integrity ]

## You should implement the functions with pass as body.
## You can define new function(s) if it helps you decompose the problem
## into smaller subproblems.

import random
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, pi
import statistics
from statistics import mean 


# Task 1

def create_array():
    """ returns created array """
    array_1D = [0] * 10  # Initialize array with zeros
    array_1D[0] = 1.0  
    array_1D[-1] = 1.0 
    return array_1D
# Task 2
def nearest_neighbours_1D(array, index):
    """ 
   Args:
       array (list): The 1D array.
       index (int): The index of the element for which neighboring elements are to be returned.

   Returns:
       list: A list containing the neighboring elements of the element at the given index.
   """
    last_index = len(array) - 1 
    if index == 0:  # Check if index is the first element
        list_start = [array[index], array[index+1]]  # Neighbors of the first element
        return list_start
    elif index == last_index: 
        list_end = [array[index-1], array[index]]  # Neighbors of the last element
        return list_end
    elif index > 0 and index < last_index:  
        list_middle = [array[index-1], array[index], array[index+1]]  # Neighbors of the middle element
        return list_middle
    else:  
        raise ValueError("Index is out of bounds")

def simulate_1d_diffusion(array):
    """ argument: current array
        returns updated array """
    diffusion_array = [*array]  # Copy the input array to avoid modifying it
    for i in range(len(array)):  # Iterate over each element in the array
        diffusion_array[i] = mean(nearest_neighbours_1D(array, i))  # Calculate the mean of neighboring elements
    return diffusion_array

# Task 3

def linspace(start, stop, num):
    """
    Args:
        start (float): The starting value of the sequence.
        stop (float): The end value of the sequence.
        num (int): Number of samples to generate.

    Returns:
        list: A list of `num` evenly spaced samples from `start` to `stop`.
    """
    step = (stop - start) / (num - 1)
    return [start + step * i for i in range(num)]

def plot_temperatures(initial, new, new2):
    """parameters: initial=original array, new=after 1 iteration, new2=after 2 iterations"""
    num_values = len(initial)
    time = linspace(0.0, 9.0, num_values)
    plt.plot(time, initial)
    plt.plot(time, new)
    plt.plot(time, new2)
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Temperature Evolution over Time')
    plt.show()

# 1D diffusion exercise code:
def exercise_1D_diffusion():    
    initial_array = create_array()
    new_array1 = simulate_1d_diffusion(initial_array)
    new_array2 = simulate_1d_diffusion(new_array1)
    plot_temperatures(initial_array, new_array1, new_array2)

# Task 4

def create_grid(size=5):
    """ argument: grid division
    returns size x size 2D grid as list of list """
    grid = []
    for _ in range(size):
        row = [0.0] * size
        grid.append(row)
    
    # Fill the border with 1.0
    for i in range(size):
        grid[0][i] = 1.0
        grid[size - 1][i] = 1.0
        grid[i][0] = 1.0
        grid[i][size - 1] = 1.0
    
    return grid

# Task 5

def nearest_neighbours_2D(array, row, column):
    """
    Args:
        array (list of lists): The 2D array.
        row (int): The row index of the element for which neighboring elements are to be returned.
        column (int): The column index of the element for which neighboring elements are to be returned.

    Returns:
        list: A list containing the neighboring elements of the element at the specified row and column. 
    """
    rows = len(array)  # Number of rows in the array
    columns = len(array[0])  # Number of columns in the array
    
    if row < 0 or row >= rows:
        raise ValueError(f"Row index must be within the range [0, {rows-1}].")

    if column < 0 or column >= columns:
        raise ValueError(f"Column index must be within the range [0, {columns-1}].")
    
    neighbours = [array[row][column]]  # Current element is added to the list of neighbours
    
    if row > 0 :
        neighbours.append(array[row-1][column])
    
    if row < rows - 1:
        neighbours.append(array[row+1][column])
    
    if column > 0:
        neighbours.append(array[row][column-1])
    
    if column < columns - 1:
        neighbours.append(array[row][column+1])
    
    return neighbours


def simulate_2d_diffusion(grid):
    """ argument: current 2D grid 
    returns updated grid after one iteration of simulation """
    diffusion_grid = [[0] * len(grid[0]) for _ in range(len(grid))]  # Initialize a new grid with the same dimensions
    rows = len(grid) 
    columns = len(grid[0])  
    for x in range(rows):  # Iterate over each row in the grid
        for y in range(columns):  # Iterate over each column in the grid
            diffusion_grid[x][y] = mean(nearest_neighbours_2D(grid, x, y))  # Calculate the mean of neighboring elements
    return diffusion_grid
   
# 2D diffusion exercise code:
def multiple_iterations(grid, num_iterations):
    for _ in range(num_iterations):
        for row in grid:
            print(' '.join(f'{temp:.2f}' for temp in row))
        print()
        grid = simulate_2d_diffusion(grid)

def exercise_2D_diffusion():    
    multiple_iterations(create_grid(),5)

# Task 6

def create_grid_numpy(size=5):
    """ argument: grid size
    returns size x size 2D grid as list of list with a padded layer of required elements"""
    array_2D = np.zeros(shape=(size-1, size-1), dtype=float)  # Create a 2D array of zeros with the specified size
    array_2D = np.pad(array_2D, pad_width=1, mode='constant', constant_values=1.0)  # Add a padded layer of boundary elements
    return array_2D


def simulate_2d_diffusion_numpy(grid):
    """ argument: current 2D grid 
    returns updated grid after one iteration of simulation """
    grid = np.array(grid) 
    diffusion_array_2d = np.copy(grid).astype(float)  # Create a copy of the grid
    rows, columns = grid.shape  # Get the number of rows and columns in the grid
    for x in range(rows):  
        for y in range(columns):  
            diffusion_array_2d[x][y] = np.mean(nearest_neighbours_2D(grid, x, y))  # Calculate the mean of neighboring elements
    return diffusion_array_2d  

def simulate_large_scale(num_iterations,size=10):
    """ arguments: num_iterations=number of iterations to perform,
                   size=dimension of 2D grid 
        No return value.
        Use NumPy for efficient large-scale simulation and visualization, correctly handling edges."""
    array = create_grid_numpy(size)
    
    for _ in range(num_iterations):
        array = simulate_2d_diffusion_numpy(array)
    # Plotting the heatmap
    plt.imshow(array, cmap='hot', interpolation='nearest')
    # Add color bar
    plt.colorbar()
    # Add title
    plt.title('Tempurature Distribution After Iteration')
    plt.show()

# 2D diffusion (numpy implementation) exercise code:
def exercise_2D_diffusion_numpy():    
    simulate_large_scale(5)
    
# Task 7:
    
def create_graph():
    """Generates a graph with nodes having initial random temperatures stored in a separate list."""
    num_nodes = 10
    # Initialize node temperatures
    temperatures = [random.randint(20, 30) for _ in range(num_nodes)]
    # Adjacency list to store edges
    edges = [[] for _ in range(num_nodes)]
    # Manually adding edges
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (0, 9),
                   (1, 3), (2, 4), (5, 7), (6, 8), (0, 5)]
    for start, end in connections:
        edges[start].append(end)
        edges[end].append(start)
    return edges, temperatures

def visualize_graph(edges, temperatures):
    """Visualizes the graph with node labels showing temperatures."""
    plt.figure(figsize=(10, 10))  # Increase figure size for better visibility
    num_nodes = len(temperatures)
    # Position nodes in a circle
    positions = {i: (cos(2 * pi * i / num_nodes), sin(2 * pi * i / num_nodes)) for i in range(num_nodes)}
    # Draw edges
    for i, neighbors in enumerate(edges):
        for neighbor in neighbors:
            plt.plot([positions[i][0], positions[neighbor][0]], 
                     [positions[i][1], positions[neighbor][1]], 'gray')
    # Draw nodes larger and with clear labels
    for i, pos in positions.items():
        plt.scatter(*pos, color='lightblue', s=1000)  # Increased node size
        plt.text(pos[0], pos[1], f'{temperatures[i]:.1f}°C', color='black', ha='center', va='center', fontweight='bold', fontsize=10)
    plt.axis('off')
    plt.show()

def simulate_diffusion(edges, temperatures):
    """ arguments: edges=edge list defining graph, 
    temperatures=current temps of graph nodes
    returns updated temperatures list"""
    num_nodes = len(temperatures)  
    updated_temperatures = temperatures[:]  # Create a copy of the temperatures list
    for node in range(num_nodes):  
        neighbours = edges[node]  
        if neighbours:  # Check if the current node has neighbors
            neighbour_temperatures = [temperatures[neighbour] for neighbour in neighbours]  
            mean_temperature = statistics.mean(neighbour_temperatures)  
            updated_temperatures[node] = mean_temperature  # Update the temperature of the current node
    return updated_temperatures  
# Graph diffusion exercise code:

def exercise_graph_diffusion():
    edges, temperatures = create_graph()
    print("Initial temperatures:", temperatures)
    visualize_graph(edges, temperatures)
    for _ in range(3):  # simulate multiple iterations
        temperatures = simulate_diffusion(edges, temperatures)
        visualize_graph(edges, temperatures)

    