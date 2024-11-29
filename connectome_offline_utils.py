import numpy as np
from tqdm import tqdm

from connectome import Connectome


def calculate_synapse_dist_to_post_syn_soma(neurons: Connectome.NeuronsDict):
    """
    """
    for neuron in tqdm(neurons.values()):
        sk = neuron.load_skeleton()
        if sk is None:
            continue

        all_syn_xyz = [syn.center_position * np.array([4, 4, 40]) for syn in neuron.pre_synapses]
        syn_ds_to_nodes, syn_nodes = sk.kdtree.query(all_syn_xyz)
        distances_to_soma = [sk.distance_to_root[node] for node in syn_nodes]
        list(map(lambda syn, dist: setattr(syn, 'dist_to_post_syn_soma', dist),
                 neuron.pre_synapses, distances_to_soma))
