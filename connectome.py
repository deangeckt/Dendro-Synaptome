import pickle
from typing import TypedDict

import numpy as np
from tqdm import tqdm

from connectome_types import SynapseDirection, cell_types, CONNECTOME_BASE_PATH, m_types, CONNECTOME_TOY_PATH
from neuron import Neuron
from synapse import Synapse

NeuronsDict = dict[int, Neuron]


class ConnectomeDict(TypedDict):
    neurons: NeuronsDict
    synapses: list[Synapse]


class Connectome:
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

    def get_cell_type_distribution(self,
                                   cell_type: str,
                                   type_space: list[str],
                                   direction: SynapseDirection,
                                   ) -> dict:
        """
        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :param direction: SynapseDirection (input, output)
        :return: cell type distribution based on the given direction
        """
        data = {type_: [] for type_ in type_space}
        for syn in tqdm(self.synapses):
            pre_syn_neuron_type = getattr(self.neurons[syn.pre_pt_root_id], cell_type)
            post_syn_neuron_type = getattr(self.neurons[syn.post_pt_root_id], cell_type)
            neuron_type = pre_syn_neuron_type if direction == SynapseDirection.output else post_syn_neuron_type
            data[neuron_type] += 1
        return data

    def get_cell_type_synapse_attr(self,
                                   cell_type: str,
                                   type_space: list[str],
                                   direction: SynapseDirection,
                                   syn_attr: str,
                                   ) -> dict:
        """
        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :param direction: SynapseDirection (input, output)
        :param syn_attr: str: (size, dist_to_post_syn_soma)
        :return: data represented in a dict, where each key is a cell_type, and the value are list of all
        aggregated values of the given calculated attribute in a tuple format: (syn_id, attr).
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

            if direction == SynapseDirection.output:
                synapse_attributes[pre_syn_neuron_type].append(syn_data)
            else:
                synapse_attributes[post_syn_neuron_type].append(syn_data)

        return synapse_attributes


if __name__ == "__main__":
    connectome = Connectome()
    # TODO: fix
    # mtype_dist_in = connectome.get_cell_type_distribution('mtype', m_types, SynapseDirection.input)

    # d = connectome.get_cell_type_synapse_attr('cell_type',
    #                                           cell_types,
    #                                           SynapseDirection.output,
    #                                           'dist_to_post_syn_soma')
    # print(d)
