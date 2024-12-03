import pickle
from typing import TypedDict

import numpy as np
from tqdm import tqdm

from connectome_types import SynapseSide, cell_types, CONNECTOME_BASE_PATH, m_types, CONNECTOME_TOY_PATH, ClfType
from neuron import Neuron
from synapse import Synapse

NeuronsDict = dict[int, Neuron]


class ConnectomeDict(TypedDict):
    neurons: NeuronsDict
    synapses: list[Synapse]


class Connectome:
    """
    This Connectome represents a small partial part of the whole MICrONS Dataset, It contains
    all neurons, but the synapses are those that each endpoint (i.e., the post/pre-synaptic neuron) belongs
    to the dataset, in other words, synapses coming from (pre-synaptic) neurons outside the EM volume are excluded,
    resulting in ~13M synapses instead of ~300M+.
    """

    def __init__(self):
        with open(CONNECTOME_TOY_PATH, 'rb') as f:
            connectome_dict: ConnectomeDict = pickle.load(f)
            self.neurons: NeuronsDict = connectome_dict['neurons']
            self.synapses: list[Synapse] = connectome_dict['synapses']

            print('Connectome:')
            print(f'\t#neurons: {len(self.neurons.keys())}')
            print(f'\t#synapses: {len(self.synapses)}')

    def get_cell_type_conn_matrix(self, cell_type: str, type_space: list[str]) -> np.ndarray:
        """
        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :return: connectivity matrix of type cell_type
        """
        conn_matrix = np.zeros((len(type_space), len(type_space)), dtype=int)
        type_index = {t: i for i, t in enumerate(type_space)}

        for syn in tqdm(self.synapses):
            pre_syn_neuron_type = getattr(self.neurons[syn.pre_pt_root_id], cell_type)
            post_syn_neuron_type = getattr(self.neurons[syn.post_pt_root_id], cell_type)
            conn_matrix[type_index[pre_syn_neuron_type], type_index[post_syn_neuron_type]] += 1

        return conn_matrix

    def get_degree_distribution(self, side: SynapseSide) -> list[int]:
        """
        Get the full degree distribution of the connectome
        i.e.: include synapses from outside the EM volume as well
        :param side: SynapseSide (pre, post)
        :return: a list of degrees
        """
        if side == SynapseSide.pre:
            return [n.num_of_ds_pre_synapses for n in self.neurons.values()]
        else:
            return [n.num_of_ds_post_synapses for n in self.neurons.values()]

    def get_cell_type_degree_distribution(self, cell_type: str, type_space: list[str], side: SynapseSide) -> dict:
        """
        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :param side: SynapseSide (pre, post)
        :return: Distribution based on the cell_type of pre- / post-neurons (based on the side)
        """
        degree_dist_per_type = {type_: [] for type_ in type_space}
        neuron_side = 'num_of_ds_pre_synapses' if side == SynapseSide.pre else 'num_of_ds_post_synapses'

        for n in tqdm(self.neurons.values()):
            degree_dist_per_type[getattr(n, cell_type)].append(getattr(n, neuron_side))

        return degree_dist_per_type

    def get_cell_type_synapse_attr(self, cell_type: str, type_space: list[str], side: SynapseSide,
                                   syn_attr: str) -> dict:
        """
        get synapses attributes divided into cell type.
        Notice - this includes synapses that both sides are within the EM volume!
        # TODO: in another file, can calculate the same for the whole EM, just for outgoing (loop over neurons).

        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :param side: SynapseSide (pre, post)
        :param syn_attr: str: (size, dist_to_post_syn_soma)
        :return: data represented in a dict, where each key is a cell_type, and the value are list of all
        aggregated values of the given synapse attribute in a tuple format: (syn_id, attr).
        """
        synapse_attributes = {type_: [] for type_ in type_space}

        for syn in tqdm(self.synapses):
            pre_syn_neuron_type = getattr(self.neurons[syn.pre_pt_root_id], cell_type)
            post_syn_neuron_type = getattr(self.neurons[syn.post_pt_root_id], cell_type)

            # For some neurons the skeleton file is corrupted
            if syn_attr == 'dist_to_post_syn_soma':
                if not hasattr(syn, 'dist_to_post_syn_soma') or syn.dist_to_post_syn_soma == -1.0:
                    continue

            syn_attr_data = getattr(syn, syn_attr)
            syn_data = (syn.id_, syn_attr_data)

            if side == SynapseSide.pre:
                synapse_attributes[pre_syn_neuron_type].append(syn_data)
            else:
                synapse_attributes[post_syn_neuron_type].append(syn_data)

        return synapse_attributes


if __name__ == "__main__":
    connectome = Connectome()
    print(connectome.get_cell_type_degree_distribution('mtype', m_types, SynapseSide.post))
    # print(connectome.get_neuron_degree_distribution('mtype', m_types, SynapseSide.post))
    # print(connectome.get_neuron_degree_distribution('mtype', m_types, SynapseSide.pre))
    #
    # clf_type_space = [e for e in ClfType]
    # print(connectome.get_neuron_degree_distribution('clf_type', clf_type_space, SynapseSide.pre))
    # print(connectome.get_neuron_degree_distribution('clf_type', clf_type_space, SynapseSide.post))
