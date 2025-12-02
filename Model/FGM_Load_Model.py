import numpy as np
import pandas as pd

def square(grid, center_x, center_y, side_length_km, height, density, node_size_km):
    """
    Apply a square load block to the grid.

    Parameters
    ----------
    grid : pandas.DataFrame
        Grid containing 'x', 'y', 'load', and 'thickness' columns.
    center_x, center_y : float
        Coordinates of the block center (meters).
    side_length_km : float
        Side length of the square block (kilometers).
    height : float
        Thickness of the block (meters).
    density : float
        Density of the block material (kg/m^3).
    node_size_km : float
        Node size in kilometers (used for scaling load contribution).

    Returns
    -------
    None
        Modifies `grid` in place by updating:
        - 'load' : applied load contribution (N).
        - 'thickness' : block thickness at each node (m).

    Notes
    -----
    - Load is computed as thickness × density × volume scaling.
    - Only nodes within the square footprint receive nonzero thickness.
    """
    side_length_m = side_length_km * 1000
    half_side = side_length_m / 2
    x_min = center_x - half_side
    x_max = center_x + half_side
    y_min = center_y - half_side
    y_max = center_y + half_side

    block_thickness = np.where(
        (grid['x'] >= x_min) & (grid['x'] <= x_max) &
        (grid['y'] >= y_min) & (grid['y'] <= y_max),
        height, 0
    )

    grid['load'] += block_thickness * density * 1000**3 * node_size_km
    grid['thickness'] = block_thickness


def circular(grid, center_x, center_y, radius_km, height, density, node_size_km):
    """
    Apply a circular load block to the grid.

    Parameters
    ----------
    grid : pandas.DataFrame
        Grid containing 'x', 'y', 'load', and 'thickness' columns.
    center_x, center_y : float
        Coordinates of the block center (meters).
    radius_km : float
        Radius of the circular block (kilometers).
    height : float
        Thickness of the block (meters).
    density : float
        Density of the block material (kg/m^3).
    node_size_km : float
        Node size in kilometers (used for scaling load contribution).

    Returns
    -------
    None
        Modifies `grid` in place by updating:
        - 'load' : applied load contribution (N).
        - 'thickness' : block thickness at each node (m).

    Notes
    -----
    - Load is computed as thickness × density × volume scaling.
    - Only nodes within the circular footprint receive nonzero thickness.
    """
    radius_m = radius_km * 1000
    dist = np.hypot(grid['x'] - center_x, grid['y'] - center_y)

    block_thickness = np.where(dist <= radius_m, height, 0)

    grid['load'] += block_thickness * density * 1000**3 * node_size_km
    grid['thickness'] = block_thickness


def load_model_builder(grid, center_x, center_y, height, node_size_km,
                       shape, density, side_length=None, radius=None):
    """
    Build a load model on the grid by applying either a square or circular block.

    Parameters
    ----------
    grid : pandas.DataFrame
        Grid containing 'x', 'y' coordinates. Will be modified in place.
    center_x, center_y : float
        Coordinates of the block center (meters).
    height : float
        Thickness of the block (meters).
    node_size_km : float
        Node size in kilometers (used for scaling load contribution).
    shape : {'square', 'circle'}
        Shape of the load block to apply.
    density : float
        Density of the block material (kg/m^3).
    side_length : float, optional
        Side length of square block (kilometers). Required if shape='square'.
    radius : float, optional
        Radius of circular block (kilometers). Required if shape='circle'.

    Returns
    -------
    None
        Modifies `grid` in place by adding:
        - 'load' : applied load contribution (N).
        - 'thickness' : block thickness at each node (m).

    Raises
    ------
    ValueError
        If required parameters for the chosen shape are missing,
        or if an unsupported shape is provided.

    Notes
    -----
    - Initializes 'load' and 'thickness' columns to zero before applying block.
    - Calls `square()` or `circular()` depending on shape.
    """
    grid['load'] = np.zeros(len(grid))
    grid['thickness'] = np.zeros(len(grid))

    if shape == 'square':
        if side_length is None:
            raise ValueError("side_length must be provided for square shape")
        square(grid, center_x, center_y, side_length, height, density, node_size_km)

    elif shape == 'circle':
        if radius is None:
            raise ValueError("radius must be provided for circle shape")
        circular(grid, center_x, center_y, radius, height, density, node_size_km)

    else:
        raise ValueError(f"Unsupported shape: {shape}")