import os
import pickle

import numpy as np

from connectome_types import EM_NEURONS_PATH, SKELETONS_DIR_PATH
from neuron import Neuron


def validate_neurons_files_and_skeletons():
    """
    Validate that the skeleton files are the same as the raw neuron files
    """
    neuron_files = os.listdir(EM_NEURONS_PATH)
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


def calculate_synapse_dist_to_soma(neuron: Neuron):
    """
    calculated the distances of each of the neurons' synapses to its soma,
    overriding the neuron incoming (pre) / outgoing (post) synapses list object
    :param neuron: a neuron object
    """
    try:
        sk = neuron.load_skeleton()
        if sk is None:
            return

        post_xyz = [syn.center_position * np.array([4, 4, 40]) for syn in neuron.post_synapses]
        pre_xyz = [syn.center_position * np.array([4, 4, 40]) for syn in neuron.pre_synapses]

        if post_xyz:
            _, syn_nodes = sk.kdtree.query(post_xyz)
            distances_to_soma = [sk.distance_to_root[node] for node in syn_nodes]
            list(map(lambda syn, dist: setattr(syn, 'dist_to_pre_syn_soma', dist),
                     neuron.post_synapses, distances_to_soma))

        if pre_xyz:
            _, syn_nodes = sk.kdtree.query(pre_xyz)
            distances_to_soma = [sk.distance_to_root[node] for node in syn_nodes]
            list(map(lambda syn, dist: setattr(syn, 'dist_to_post_syn_soma', dist),
                     neuron.pre_synapses, distances_to_soma))

    except Exception as e:
        print(e)
        print(f'calc_syn_dist failed for neuron: {neuron.root_id}')


def __compute_depth(sk, node, depth_cache):
    if node in depth_cache:
        return depth_cache[node]

    if node == int(sk.root):
        depth_cache[node] = 0
        return 0

    syn_segment = sk.segment_map[node]
    parent_node = sk.segments_plus[syn_segment][-1]

    depth = 1 + __compute_depth(sk, parent_node, depth_cache)
    depth_cache[node] = depth
    return depth


def calculate_synapse_depth(neuron: Neuron):
    """
    calculated the depth of each of the neurons' (pre) synapses, over the branching tree,
    overriding the neuron pre_synapses list object
    :param neuron: a neuron object
    """
    try:
        sk = neuron.load_skeleton()
        if sk is None:
            return

        depth_cache = {}

        post_xyz = [syn.center_position * np.array([4, 4, 40]) for syn in neuron.post_synapses]
        _, syn_nodes = sk.kdtree.query(post_xyz)
        for node, syn in zip(post_xyz, neuron.post_synapses):
            syn.depth_in_pre_syn_tree = __compute_depth(sk=sk, node=node, depth_cache=depth_cache)

        pre_xyz = [syn.center_position * np.array([4, 4, 40]) for syn in neuron.pre_synapses]
        _, syn_nodes = sk.kdtree.query(pre_xyz)
        for node, syn in zip(syn_nodes, neuron.pre_synapses):
            syn.depth_in_post_syn_tree = __compute_depth(sk=sk, node=node, depth_cache=depth_cache)

    except Exception as e:
        print(e)
        print(f'calculate_synapse_depth failed for neuron: {neuron.root_id}')


if __name__ == "__main__":
    failed_sk = {864691134965932575,
                 864691135183340034,
                 864691135368655609,
                 864691135395659765,
                 864691135395662581,
                 864691135430150448,
                 864691135430156592,
                 864691135430169648,
                 864691135447917396,
                 864691135479759814,
                 864691135479767238,
                 864691135492640607,
                 864691135503317597,
                 864691135570667142,
                 864691135610942983,
                 864691135688499936,
                 864691135716530202,
                 864691135731801017,
                 864691135731803321,
                 864691135782381392,
                 864691135816656207,
                 864691135860081768,
                 864691135860136552,
                 864691135860313960,
                 864691135876855379,
                 864691135919820464,
                 864691135926300942,
                 864691135927205588,
                 864691135927205844,
                 864691135927217364,
                 864691135927218388,
                 864691135927342292,
                 864691135927571156,
                 864691135939327489,
                 864691135939329025,
                 864691135940871846,
                 864691135945542180,
                 864691135945550116,
                 864691135945550628,
                 864691135945551908,
                 864691135945553700,
                 864691135945562660,
                 864691135945705508,
                 864691135945715236,
                 864691135945888548,
                 864691135945987620,
                 864691135953453603,
                 864691135974872687,
                 864691135974883183,
                 864691135974987631,
                 864691135974987887,
                 864691135974992495,
                 864691135975213679,
                 864691135975235439,
                 864691135977176643,
                 864691135993214913,
                 864691136006200522,
                 864691136021867000,
                 864691136085637996,
                 864691136335250355,
                 864691136391836799,
                 864691136579242516,
                 864691136618482317,
                 864691136674341255,
                 864691136674342535,
                 864691136926769226,
                 864691136926770250}
    # unable to download (and load) the skeleton
    for n in list(failed_sk):
        with open(os.path.join(EM_NEURONS_PATH, f'{n}.pkl'), 'rb') as f:
            neuron: Neuron = pickle.load(f)
            calculate_synapse_depth(neuron)

    # client = CAVEclient('minnie65_public')
    # for cell_id in failed_sk:
    #     sk_dict = client.skeleton.get_skeleton(cell_id, output_format='json')
    #     print(sk_dict)
