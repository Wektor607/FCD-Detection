import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import nibabel as nb
import copy
import time
from scipy import sparse
# import meld_classifier.mesh_tools as mt
import torch
from math import pi
import logging
from meld_graph.icospheres import IcoSpheres
import matplotlib_surface_plotting as msp
from scipy.stats import special_ortho_group
from scipy.spatial import cKDTree
import pickle
from meld_graph.resampling_meshes import *


if __name__ == "__main__":
    # initialise params
    for ico_index in range(1, 8):
    # ico_index = 7
        flip_axes = 3
        data_dir = "../data/flipping"
        file_name = f"flipping_ico{ico_index}_" + "{}.npy"

        # Create icospheres
        icos = IcoSpheres(conv_type="SpiralConv")

        # Create multiple spinned icosphere
        flipped_lambdas = []
        flipped_indices = []
        for ax in range(0, flip_axes):
            # Warp icosphere at second level

            flipped_coords = copy.deepcopy(icos.icospheres[ico_index]["coords"])
            flipped_coords[:, ax] = -flipped_coords[:, ax]

            # Find nearest 3 neighbours vertices from spinned ico for each vertices in initial ico
            tree = cKDTree(flipped_coords)
            distance, indices = tree.query(icos.icospheres[ico_index]["coords"], k=3)
            lambdas = barycentric_coordinates_matrix(icos.icospheres[ico_index]["coords"], flipped_coords[indices])
            redos = np.where(~np.logical_and(0 < lambdas, lambdas < 1).all(axis=1))[0]
            # fix the ones that aren't quite right
            indices, lambdas = correct_triangles(icos.icospheres[ico_index], indices, redos, flipped_coords, lambdas)
            # Add to multiple arrays
            flipped_lambdas.append(lambdas)
            flipped_indices.append(indices)
        flipped_lambdas = np.array(flipped_lambdas)
        flipped_indices = np.array(flipped_indices)

        # Save dictionary with lambdas and indices
        output_file = os.path.join(data_dir, file_name.format(flip_axes))
        data = (flipped_lambdas, flipped_indices)
        np.save(output_file, data)
