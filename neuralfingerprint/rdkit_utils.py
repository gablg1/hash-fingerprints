import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import autograd.numpy as np
from features import num_atom_features, num_bond_features
from util import memoize, WeightsParser
from mol_graph import graph_from_smiles_tuple, degrees
import random

TABLE_SIZE = 100000
K = 5

def hash_function(x, cache):
    if x not in cache:
        cache[x] = random.randint(0, TABLE_SIZE - 1)
    return cache[x]

hash_caches = [{} for _ in xrange(K)]
hash_functions = [lambda x, c=cache: hash_function(x, c) for cache in hash_caches]

def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    molgraph = graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep

def smiles_to_myfps(data, fp_length, fp_radius):
    #return np.array([smile_to_myfpp(s, fp_length, fp_radius) for s in data]

    array_rep = array_rep_from_smiles(tuple(data))
    fps = [smile_to_bloom_fp(atom_list, array_rep, fp_length, fp_radius) for atom_list in array_rep['atom_list']]
    return np.array(fps)

def smile_to_myfpp(s, fp_length, fp_radius):
    molgraph = graph_from_smiles_tuple(smiles)

    fp = [0 for _ in range(fp_length)]
    for degree in xrange(fp_radius):
        for a in atom_list:
            #neighbor_features = neighbors(a, l)
            #v = np.array([atom_features[a]] + neighbor_features)
            neighbor_features = []
            if len(atom_neighbors_list) > 0:
                for neighbor in atom_neighbors_list[a]:
                    neighbor_features.append(atom_features[neighbor])
                for bond in bond_neighbors_list[a]:
                    neighbor_features.append(bond_features[bond])
            v = atom_features[a]
            for hash_fn in hash_functions:
                idx = hash_function(tuple(v)) % fp_length
                fp[idx] = 1
    return fp

def smile_to_myfp(atom_list, array_rep, fp_length, fp_radius):
    atom_features = array_rep['atom_features']
    bond_features = array_rep['bond_features']

    fp = [0 for _ in range(fp_length)]
    for degree in xrange(fp_radius):
        atom_neighbors_list = array_rep[('atom_neighbors', degree)]
        bond_neighbors_list = array_rep[('bond_neighbors', degree)]

        for a in atom_list:
            #neighbor_features = neighbors(a, l)
            #v = np.array([atom_features[a]] + neighbor_features)
            neighbor_features = []
            v = atom_features[a]
            idx = hash(tuple(v)) % fp_length
            fp[idx] = 1
    return fp

def smile_to_bloom_fp(atom_list, array_rep, fp_length, fp_radius):
    atom_features = array_rep['atom_features']
    bond_features = array_rep['bond_features']

    fp = [0 for _ in range(fp_length)]
    for degree in xrange(fp_radius):
        atom_neighbors_list = array_rep[('atom_neighbors', degree)]
        bond_neighbors_list = array_rep[('bond_neighbors', degree)]

        for a in atom_list:
            #neighbor_features = neighbors(a, l)
            #v = np.array([atom_features[a]] + neighbor_features)

            neighbor_features = []
            v = atom_features[a]
            idx = hash_functions[0](tuple(v)) % fp_length
            fp[idx] = 1
    return fp

def smiles_to_fps(data, fp_length, fp_radius):
    return stringlist2intarray(np.array([smile_to_fp(s, fp_length, fp_radius) for s in data]))

def smile_to_fp(s, fp_length, fp_radius):
    m = Chem.MolFromSmiles(s)
    return (AllChem.GetMorganFingerprintAsBitVect(
        m, fp_radius, nBits=fp_length)).ToBitString()

def stringlist2intarray(A):
    '''This function will convert from a list of strings "10010101" into in integer numpy array.'''
    return np.array([list(s) for s in A], dtype=int)
