import os

import numpy as np
from tqdm import tqdm

from connectome import NeuronsDict
from connectome_types import NEURONS_PATH, SKELETONS_DIR_PATH


def validate_neurons_files_and_skeletons():
    neuron_files = os.listdir(NEURONS_PATH)
    neuron_files = [n.split('.')[0] for n in neuron_files]
    neuron_files = set(neuron_files)

    sk_files = os.listdir(SKELETONS_DIR_PATH)
    sk_files = [n.split('.')[0] for n in sk_files]
    sk_files = set(sk_files)

    print(f'# neuron files {len(neuron_files)}')
    print(f'# skeleton files {len(sk_files)}')

    assert sk_files.symmetric_difference(neuron_files) == {'864691135738308740', '864691135293327798',
                                                           '864691135490593127', '864691135491249759',
                                                           '864691135515916499', '864691135527121243'}


def calculate_synapse_dist_to_post_syn_soma(neurons: NeuronsDict):
    for neuron in tqdm(neurons.values()):
        sk = neuron.load_skeleton()
        if sk is None:
            continue

        all_syn_xyz = [syn.center_position * np.array([4, 4, 40]) for syn in neuron.pre_synapses]
        if not all_syn_xyz:
            return
        syn_ds_to_nodes, syn_nodes = sk.kdtree.query(all_syn_xyz)
        distances_to_soma = [sk.distance_to_root[node] for node in syn_nodes]
        list(map(lambda syn, dist: setattr(syn, 'dist_to_post_syn_soma', dist),
                 neuron.pre_synapses, distances_to_soma))
