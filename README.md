# Diffusion Simulation Project

## Overview
This project implements simulations of diffusion processes in 1D arrays, 2D grids, and graphs using Python. The simulations demonstrate how values evolve over time based on their neighbors, modeled with statistical means. Visualization of results is provided using `Matplotlib`, while large-scale computations leverage the efficiency of `NumPy`.

## Key Features
1. **1D Diffusion Simulation:**
   - Implements diffusion on a one-dimensional array.
   - Visualizes the evolution of the array over time using line plots.

2. **2D Diffusion Simulation:**
   - Simulates diffusion on a two-dimensional grid.
   - Supports both pure Python and `NumPy` implementations for performance optimization.
   - Displays heatmaps to visualize temperature distributions.

3. **Graph-Based Diffusion Simulation:**
   - Creates and visualizes a graph with nodes and edges.
   - Simulates diffusion where node values are influenced by connected neighbors.
   - Provides graphical representations of the diffusion process.

## Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - `Matplotlib` for data visualization.
  - `NumPy` for efficient numerical computations.
  - `Statistics` module for mean calculations.
  - `Math` for trigonometric operations in graph visualization.
- **Random Module:** For generating initial temperatures in graph-based diffusion.

## Functions Overview

### 1D Diffusion
- **`create_array()`**: Creates a 1D array with boundary values set to 1.0.
- **`nearest_neighbours_1D(array, index)`**: Returns the neighboring values of a given index.
- **`simulate_1d_diffusion(array)`**: Updates the array based on the average of neighboring elements.
- **`plot_temperatures(initial, new, new2)`**: Plots the temperature evolution over time.
- **`exercise_1D_diffusion()`**: Demonstrates the 1D diffusion simulation.

### 2D Diffusion
- **`create_grid(size)`**: Creates a 2D grid with boundary values set to 1.0.
- **`nearest_neighbours_2D(array, row, column)`**: Returns neighboring values of a grid element.
- **`simulate_2d_diffusion(grid)`**: Updates the grid based on the average of neighboring elements.
- **`create_grid_numpy(size)`**: Creates a 2D grid using `NumPy`.
- **`simulate_2d_diffusion_numpy(grid)`**: Efficiently updates the grid using `NumPy`.
- **`simulate_large_scale(num_iterations, size)`**: Simulates diffusion on a larger grid and visualizes results with a heatmap.
- **`exercise_2D_diffusion()`**: Demonstrates the 2D diffusion simulation.
- **`exercise_2D_diffusion_numpy()`**: Demonstrates the `NumPy`-optimized 2D diffusion simulation.

### Graph-Based Diffusion
- **`create_graph()`**: Generates a graph with nodes and initial random temperatures.
- **`visualize_graph(edges, temperatures)`**: Visualizes the graph with node temperatures.
- **`simulate_diffusion(edges, temperatures)`**: Simulates diffusion based on graph edges and node temperatures.
- **`exercise_graph_diffusion()`**: Demonstrates the graph-based diffusion simulation.

## Installation and Requirements
1. Install Python 3.x.
2. Install the required libraries using pip:
   ```bash
   pip install matplotlib numpy
   ```

## Usage
1. Clone the repository or copy the script.
2. Run the desired exercise function to see diffusion simulations:
   - **1D Diffusion**: `exercise_1D_diffusion()`
   - **2D Diffusion (Pure Python)**: `exercise_2D_diffusion()`
   - **2D Diffusion (NumPy)**: `exercise_2D_diffusion_numpy()`
   - **Graph-Based Diffusion**: `exercise_graph_diffusion()`
3. View the visualizations generated by the functions.

## Examples
### 1D Diffusion
A 1D array evolves over two iterations, visualized as line plots showing the progression of values over time.

### 2D Diffusion
A 5x5 grid demonstrates the temperature spread, shown as a heatmap after multiple iterations.

### Graph Diffusion
A graph of 10 nodes visualizes the diffusion of temperatures based on connected neighbors.

## License
This project is licensed under the MIT License.

---

## Acknowledgment
This project is part of the COMP1730/6730 course at ANU. All work is original and adheres to the university’s academic integrity guidelines.

---

For more information, feel free to contact **Manav Singh** at u7782612.