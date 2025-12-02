import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.special import kei

def grid_nodes(node_size, grid_size):
    """
    Create a square grid of nodes with spatial coordinates.

    Parameters
    ----------
    node_size : float
        Size of each node in kilometers (side length of square).
    grid_size : int
        Number of nodes along each axis. If even, incremented to odd.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - 'nx', 'ny' : integer node indices
        - 'x', 'y'   : spatial coordinates in meters

    Notes
    -----
    - Coordinates are centered around (0,0).
    - Node spacing is `node_size * 1000` meters.
    """
    if grid_size % 2 == 0:
        grid_size += 1  

    start = -(grid_size-1)/2
    stop = (grid_size-1)/2+1
    offset = node_size*1000
    boxs = np.arange(start, stop)
    coords = np.arange(start, stop) * offset

    node_grid = [(int(x), int(y)) for y in boxs for x in boxs]
    node_coords = [(int(x), int(y)) for y in coords for x in coords]

    rows = [(nx, ny, x, y) for (nx, ny), (x, y) in zip(node_grid, node_coords)]  
    grid = pd.DataFrame(rows, columns=['nx', 'ny','x','y'])
    return grid


def flexure_displacement(grid, x_current, y_current, load, c, l):
    """
    Apply flexural displacement at all nodes due to a point load.

    Parameters
    ----------
    grid : pandas.DataFrame
        Grid containing 'x', 'y', and 'w' columns.
    x_current, y_current : float
        Coordinates of the load center (meters).
    load : float
        Magnitude of applied load (N/m^2).
    c : float
        Flexural constant (scaling factor).
    l : float
        Flexural parameter (characteristic length, meters).

    Notes
    -----
    - Uses the Kelvin function kei() for radial displacement.
    - Adds displacement contribution to existing grid['w'].
    """
    dx = grid['x'].values - x_current
    dy = grid['y'].values - y_current
    dist = np.hypot(dx, dy)

    k = kei(dist / l)
    disp = load * k * c
    grid['w'] += disp    


def flexure(flexural_rigidity, gravitational_acceleration, density_mantle, grid):
    """
    Compute lithospheric flexure due to applied loads.

    Parameters
    ----------
    flexural_rigidity : float
        Rigidity of the lithosphere (N·m).
    gravitational_acceleration : float
        Gravitational acceleration (m/s^2).
    density_mantle : float
        Mantle density (kg/m^3).
    grid : pandas.DataFrame
        Grid containing 'x', 'y', and 'load' columns.

    Returns
    -------
    None
        Modifies grid in place by adding column 'w' (displacement in meters).

    Notes
    -----
    - Characteristic length l = (D / (ρ g))^(1/4).
    - Displacement is computed for all nonzero load nodes.
    """
    D = flexural_rigidity
    g = gravitational_acceleration
    Pm = density_mantle
    l = (D/(Pm*g))**0.25

    grid['w'] = np.zeros(len(grid))
    c = 1./(2.*np.pi*Pm*g*l**2)

    for _, row in grid[grid['load'] != 0].iterrows():
        flexure_displacement(grid, row['x'], row['y'], row['load'], c, l)  


def flexure_plotter(grid, node_size, colormap='viridis', title='Flexural Displacement (m)'):
    """
    Plot flexural displacement field using colored rectangles.

    Parameters
    ----------
    grid : pandas.DataFrame
        Grid containing 'x', 'y', and 'w' columns.
    node_size : float
        Node size in kilometers (used to scale rectangles).
    colormap : str, optional
        Matplotlib colormap (default 'viridis').
    title : str, optional
        Plot title.

    Returns
    -------
    None
        Displays two plots:
        - Map view of displacement field.
        - Cross-section along x = 0.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(grid['w'].min(), grid['w'].max())

    box_size = node_size*1000
    for _, row in grid.iterrows():
        color = cmap(norm(row['w']))
        lower_left = (row['x'] - box_size / 2, row['y'] - box_size / 2)
        rect = Rectangle(lower_left, box_size, box_size, facecolor=color)
        ax.add_patch(rect)

    ax.set_xlim(grid['x'].min(), grid['x'].max())
    ax.set_ylim(grid['y'].min(), grid['y'].max())
    ax.set_aspect('equal')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Displacement w")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    cross_section = grid[grid['x'] == 0].sort_values(by='y')
    plt.plot(cross_section['y'], cross_section['w'])
    plt.title("Cross Section of Displacement Along x = 0")
    plt.xlabel("y (m)")
    plt.ylabel("Displacement w")
    plt.grid(True)
    plt.show()


def flexure_plotter_imshow(grid, node_size, colormap='seismic', title='Flexural Displacement (m)'):
    """
    Plot flexural displacement field using imshow.

    Parameters
    ----------
    grid : pandas.DataFrame
        Grid containing 'x', 'y', and 'w' columns.
    node_size : float
        Node size in kilometers (used for extent scaling).
    colormap : str, optional
        Matplotlib colormap (default 'seismic').
    title : str, optional
        Plot title.

    Returns
    -------
    None
        Displays two plots:
        - Map view of displacement field (imshow).
        - Cross-section along x = 0.
    """
    pivot = grid.pivot(index='y', columns='x', values='w')
    x_vals = pivot.columns.values
    y_vals = pivot.index.values
    w_array = pivot.values

    max_abs = np.nanmax(np.abs(w_array))
    vmin, vmax = -max_abs, max_abs

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        w_array,
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
        origin='lower',
        cmap=cmap,
        norm=norm,
        aspect='equal'
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Displacement w")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(False)
    plt.show()

    cross_section = grid[grid['x'] == 0].sort_values(by='y')
    plt.plot(cross_section['y'], cross_section['w'])
    plt.title("Cross Section of Displacement Along x = 0")
    plt.xlabel("y (m)")
    plt.ylabel("Displacement w")
    plt.grid(True)
    plt.show()