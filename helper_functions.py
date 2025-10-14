import openmc.data
import pandas as pd
import sqlite3
import numpy as np
import os
import matplotlib.pyplot as plt

def get_A(target_symbol):
    target = target_symbol.split("-")
    try:
        A = np.int16(target[1])
    except:
        A = -1
    return A


def get_element(target_symbol):
    target = target_symbol.split("-")
    element = target[0]
    return element

def extract_XS_openMC_hdf5(path, MT, interp_energies, temp='294K', do_plot=True):
    # Load neutron data from ACE (processed at 293.6 K)
    nuclide_data = openmc.data.IncidentNeutron.from_hdf5(path)
    
    MTs_to_use = []
    
    # Choose MT = 102 (n,γ)
    xs_func = nuclide_data[MT].xs[temp]

    
    
    # Evaluate cross section on an energy grid
    grid_energy = np.logspace(-5, 7, 1000)  # eV
    grid_cross_section = xs_func(grid_energy)
    interp_cross_section = xs_func(interp_energies)
    if do_plot:
        plt.loglog(grid_energy, grid_cross_section, label="ENDF")
        plt.loglog(interp_energies, interp_cross_section, ".", label="Interpolated")
        plt.xlabel('Energy (eV)')
        plt.ylabel('Cross section (barn)')
        plt.show()
    return interp_cross_section, grid_energy, grid_cross_section


def extract_XS_openMC_ace(path, MT, interp_energies, temp='294K', do_plot=False):
    # Load neutron data from ACE (processed at 293.6 K)
    nuclide_data = openmc.data.IncidentNeutron.from_ace(path)
    
    MTs_to_use = []
    
    # Choose MT = 102 (n,γ)
    xs_func = nuclide_data[MT].xs[temp]

    
    
    # Evaluate cross section on an energy grid
    grid_energy = np.logspace(-5, 7, 1000)  # eV
    grid_cross_section = xs_func(grid_energy)
    interp_cross_section = xs_func(interp_energies)
    if do_plot:
        plt.loglog(grid_energy, grid_cross_section, label="ENDF")
        plt.loglog(interp_energies, interp_cross_section, ".", label="Interpolated")
        plt.xlabel('Energy (eV)')
        plt.ylabel('Cross section (barn)')
        plt.show()
    return interp_cross_section, grid_energy, grid_cross_section




Z_MAP = {'Ac': 89, 'Ag': 47, 'Al': 13, 'Am': 95, 'Ar': 18, 'As': 33, 'At': 85, 'Au': 79, 'B': 5, 'Ba': 56, 'Be': 4,
         'Bh': 107, 'Bi': 83, 'Bk': 97, 'Br': 35, 'C': 6, 'Ca': 20, 'Cd': 48, 'Ce': 58,
         'Cf': 98, 'Cl': 17, 'Cm': 96, 'Co': 27, 'Cr': 24, 'Cs': 55, 'Cu': 29, 'Ds': 110, 'Db': 105, 'Dy': 66, 'Er': 68,
         'Es': 99, 'Eu': 63, 'F': 9, 'Fe': 26, 'Fm': 100, 'Fr': 87, 'Ga': 31, 'Gd':
             64, 'Ge': 32, 'H': 1, 'He': 2, 'Hf': 72, 'Hg': 80, 'Ho': 67, 'Hs': 108, 'I': 53, 'In': 49, 'Ir': 77,
         'K': 19, 'Kr': 36, 'La': 57, 'Li': 3, 'Lr': 103, 'Lu': 71, 'Md': 101, 'Mg': 12, 'Mn':
             25, 'Mo': 42, 'Mt': 109, 'N': 7, 'Na': 11, 'Nb': 41, 'Nd': 60, 'Ne': 10, 'Ni': 28, 'No': 102, 'Np': 93,
         'O': 8, 'Os': 76, 'P': 15, 'Pa': 91, 'Pb': 82, 'Pd': 46, 'Pm': 61, 'Po': 84, 'Pr':
             59, 'Pt': 78, 'Pu': 94, 'Ra': 88, 'Rb': 37, 'Re': 75, 'Rf': 104, 'Rg': 111, 'Rh': 45, 'Rn': 86, 'Ru': 44,
         'S': 16, 'Sb': 51, 'Sc': 21, 'Se': 34, 'Sg': 106, 'Si': 14, 'Sm': 62, 'Sn': 50,
         'Sr': 38, 'Ta': 73, 'Tb': 65, 'Tc': 43, 'Te': 52, 'Th': 90, 'Ti': 22, 'Tl': 81, 'Tm': 69, 'U': 92, 'V': 23,
         'W': 74, 'Xe': 54, 'Y': 39, 'Yb': 70, 'Zn': 30, 'Zr': 40}


def get_z(symbol, z_map=Z_MAP):
    try:
        z = z_map[symbol]
    except:
        z = 0
    return np.int16(z)