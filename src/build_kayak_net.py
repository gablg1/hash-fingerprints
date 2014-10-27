
import sys
import numpy as np
import numpy.random as npr
import  itertools as it
from rdkit.Chem import AllChem, MolFromSmiles
sys.path.append('../../Kayak/')
import kayak
import kayak as ky
import kayak_ops as mk
from MolGraph import *
from features import *
from load_data import load_molecules

# --- Functions working with linked object representation of molecules ---

def initialize_weights(num_hidden_features, scale):
    num_layers = len(num_hidden_features)
    num_atom_features = atom_features()
    num_edge_features = bond_features()
    num_features = [num_atom_features] + num_hidden_features
    # Initialize the weights
    np_weights = {}

    for layer in range(num_layers):
        N_prev, N_next = num_features[layer], num_features[layer + 1]
        np_weights[('self', layer)]  = scale * npr.randn(N_prev, N_next)
        np_weights[('other', layer)] = scale * npr.randn(N_prev, N_next)
        np_weights[('edge', layer)]  = scale * npr.randn(num_edge_features, N_next)
    np_weights['out'] = scale * npr.randn(num_features[-1], 1)
    return np_weights

def BuildNetFromSmiles(smile, np_weights, target):
    mol = Chem.MolFromSmiles(smile)
    graph = BuildGraphFromMolecule(mol)
    return BuildNetFromGraph(graph, np_weights, target)

def BuildGraphFromMolecule(mol):
    # Replicate the graph that RDKit produces.
    # Go on and extract features using RDKit also.
    graph = MolGraph()
    AllChem.Compute2DCoords(mol)    # Only for visualization.

    # Iterate over the atoms.
    rd_atoms = {}
    for atom in mol.GetAtoms():
        new_vert = Vertex()
        new_vert.nodes = [kayak.Inputs(atom_features(atom)[None,:])]
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        new_vert.pos = (pos.x, pos.y)
        graph.add_vert( new_vert )
        rd_atoms[atom.GetIdx()] = new_vert

    # Iterate over the bonds.
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        new_edge = Edge(rd_atoms[atom1.GetIdx()], rd_atoms[atom2.GetIdx()])
        new_edge.nodes = [kayak.Inputs(bond_features(bond)[None, :])]
        graph.add_edge(new_edge)

    return graph

def BuildNetFromGraph(graph, np_weights, target):
    # This first version just tries to emulate ECFP, with different weights on each layer

    # Dict comprehension to loop over layers and types.
    k_weights = {key: kayak.Parameter(weights) for key, weights in np_weights.iteritems()}
    # Build concatenated sets of weights
    cat_weights = {}
    for layer in it.count():
        if ('self', layer) not in k_weights:
            num_layers = layer
            break

        cur_cat = k_weights[('self', layer)]
        for num_neighbors in [1, 2, 3, 4]:
            cur_cat = kayak.Concatenate(0, cur_cat,
                                        k_weights[('other', layer)],
                                        k_weights[('edge', layer)])
            cat_weights[(layer, num_neighbors)] = cur_cat

    for layer in range(num_layers):
        # Every atom and edge is a separate Kayak Input. These inputs already live in the graph.
        for v in graph.verts:
            # Create a Differentiable node N that depends on the corresponding node in the previous layer, its edges,
            # and its neighbours.
            # First we'll concatenate all the input nodes:
            nodes_to_cat = [v.nodes[layer]]
            neighbors = zip(*v.get_neighbors()) # list of (node, edge) tuple
            num_neighbors = len(neighbors)
            for n, e in neighbors:
                nodes_to_cat.append(n.nodes[layer])
                nodes_to_cat.append(e.nodes[layer])
            cat_node = kayak.Concatenate(1, *nodes_to_cat)
            v.nodes.append(kayak.Logistic(kayak.MatMult(cat_node, cat_weights[(layer, num_neighbors)])))

        for e in graph.edges:
            e.nodes.append(kayak.Identity(e.nodes[layer]))

    # Connect everything to the fixed-size layer using some sort of max
    penultimate_nodes = [v.nodes[-1] for v in graph.verts]
    concatenated = kayak.Concatenate( 0, *penultimate_nodes)
    softmax_layer = kayak.SoftMax(concatenated, axis=0)
    output_layer = kayak.MatSum(kayak.MatElemMult(concatenated, softmax_layer), axis=0)

    # Perform a little more computation to get a single number.
    output = kayak.MatMult(output_layer, k_weights['out'])
    return kayak.L2Loss(output, kayak.Targets(target)), k_weights, output

# --- Functions working with array representation of molecules ---

def build_universal_net(num_hidden, param_scale):
    # Derived parameters
    layer_sizes = [atom_features()] + num_hidden
    k_weights = []
    def new_weights(shape):
        return new_parameters(k_weights, shape, param_scale)

    mol = new_kayak_mol_input()
    cur_atoms = mol['atom_features']
    for N_prev, N_curr in zip(layer_sizes[:-1], layer_sizes[1:]):
        w_self = new_weights((N_prev, N_curr))
        w_atom_cat = cat_weights(new_weights((N_prev, N_curr)))
        w_bond_cat = cat_weights(new_weights((bond_features(), N_curr)))
        cur_atoms = ky.Logistic(ky.MatAdd(
            ky.MatMult(cur_atoms, w_self),
            mk.NeighborMatMult(cur_atoms,
                               mol['atom_atom_neighbors'],
                               w_atom_cat),
            mk.NeighborMatMult(mol['bond_features'],
                               mol['atom_bond_neighbors'],
                               w_bond_cat)))

    output = ky.MatMult(softened_max(cur_atoms), new_weights((N_curr, 1)))
    target = ky.Targets(None)
    loss = ky.L2Loss(output, target)
    return mol, target, loss, output, k_weights

def arrayrep_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    rdkit_atoms = mol.GetAtoms()
    rdkit_bonds = mol.GetBonds()
    num_atoms, num_bonds = len(rdkit_atoms), len(rdkit_bonds)
    graph = {'atom_features'       : np.zeros((num_atoms, atom_features())),
             'bond_features'       : np.zeros((num_bonds, bond_features())),
             'bond_neighbors'      : [None] * num_bonds,
             'atom_atom_neighbors' : [[] for i in xrange(num_atoms)],
             'atom_bond_neighbors' : [[] for i in xrange(num_atoms)]}
    for atom in rdkit_atoms:
        idx = atom.GetIdx()
        graph['atom_features'][idx, :] = atom_features(atom)

    for bond in rdkit_bonds:
        idx = bond.GetIdx()
        graph['bond_features'][idx, :] = bond_features(bond)
        graph['bond_neighbors'][idx] = [bond.GetBeginAtom().GetIdx(),
                                        bond.GetEndAtom().GetIdx()]

    # No new information here, just useful representations
    for bond_idx, atom_idxs in enumerate(graph['bond_neighbors']):
        atom_A, atom_B = atom_idxs
        graph['atom_atom_neighbors'][atom_A].append(atom_B)
        graph['atom_atom_neighbors'][atom_B].append(atom_A)
        graph['atom_bond_neighbors'][atom_A].append(bond_idx)
        graph['atom_bond_neighbors'][atom_B].append(bond_idx)

    return graph

def load_new_input(input_mol, simple_graph):
    for field in input_mol:
        input_mol[field].data = simple_graph[field]

def new_kayak_mol_input():
    mol_fields = ['atom_features',
                  'bond_features',
                  'atom_atom_neighbors',
                  'atom_bond_neighbors',
                  'bond_neighbors']
    return {field : kayak.Inputs(None) for field in mol_fields}

# --- Useful functions for building nets ---
def cat_weights(w):
    cat = {1 : w}
    for i in [2, 3, 4]:
        cat[i] = ky.Concatenate(0, cat[i-1], w)

    return cat

def new_parameters(parameter_list, shape, scale=0.1):
    new = ky.Parameter(np.random.randn(*shape) * scale)
    parameter_list.append(new)
    return new

def softened_max(features):
    return ky.MatSum(ky.ElemMult(features,
                                 ky.SoftMax(features, axis=0)),
                     axis=0)
