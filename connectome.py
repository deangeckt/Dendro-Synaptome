import pickle
from typing import TypedDict

import numpy as np
import pandas
import pandas as pd
from tqdm import tqdm

from connectome_types import SynapseSide, cell_types, CONNECTOME_BASE_PATH, m_types, CONNECTOME_TOY_PATH, ClfType, \
    CONNECTOME_SYN_TABLE_PATH
from neuron import Neuron
from synapse import Synapse

NeuronsDict = dict[int, Neuron]


class ConnectomeDict(TypedDict):
    neurons: NeuronsDict
    synapses: list[Synapse]


class Connectome:
    """
    This Connectome represents a partial part of the whole MICrONS Dataset, It contains
    all neurons, but the synapses are those that each endpoint (i.e., the post/pre-synaptic neuron) belongs
    to the dataset, in other words, synapses coming from (pre-synaptic) neurons outside the EM volume are excluded,
    resulting in ~13M synapses instead of ~300M+.
    """
    def __init__(self, from_disk=True, neurons=None, synapses=None):
        if from_disk:
            with open(CONNECTOME_BASE_PATH, 'rb') as f:
                connectome_dict: ConnectomeDict = pickle.load(f)
                self.neurons: NeuronsDict = connectome_dict['neurons']
                self.synapses: list[Synapse] = connectome_dict['synapses']
        else:
            self.neurons = neurons
            self.synapses = synapses

        print('Connectome:')
        print(f'\t#neurons: {len(self.neurons.keys())}')
        print(f'\t#synapses: {len(self.synapses)}')

    def get_synapses_table(self) -> pandas.DataFrame:
        synapses_size = []
        synapses_pos = []
        synapses_dist_to_soma = []
        synapses_depth = []
        syn_id = []

        # pre-synaptic neuron data
        pre_clf_type = []
        pre_cell_type_type = []
        pre_mtype_type = []

        # post-synaptic neuron data
        post_clf_type = []
        post_cell_type = []
        post_mtype_type = []

        for syn in tqdm(self.synapses):
            syn_id.append(syn.id_)
            synapses_size.append(syn.size / 1000)
            synapses_pos.append(syn.center_position * np.array([4, 4, 40]))

            syn_depth = syn.depth if hasattr(syn, 'depth') else -1.0
            synapses_depth.append(syn_depth)

            syn_dist_to_soma = syn.dist_to_post_syn_soma if hasattr(syn, 'dist_to_post_syn_soma') else -1.0
            synapses_dist_to_soma.append(syn_dist_to_soma / 1000)

            pre_syn_neuron = self.neurons[syn.pre_pt_root_id]
            pre_clf_type.append(pre_syn_neuron.clf_type)
            pre_cell_type_type.append(pre_syn_neuron.cell_type)
            pre_mtype_type.append(pre_syn_neuron.mtype)

            post_syn_neuron = self.neurons[syn.post_pt_root_id]
            post_clf_type.append(post_syn_neuron.clf_type)
            post_cell_type.append(post_syn_neuron.cell_type)
            post_mtype_type.append(post_syn_neuron.mtype)

        return pd.DataFrame({'id_': syn_id, 'dist_to_post_syn_soma': synapses_dist_to_soma, 'size': synapses_size,
                             'center_position': synapses_pos, 'depth': synapses_depth,
                             'pre_clf_type': pre_clf_type, 'pre_cell_type': pre_cell_type_type,
                             'pre_m_type': pre_mtype_type, 'post_clf_type': post_clf_type,
                             'post_cell_type': post_cell_type, 'post_mtype_type': post_mtype_type,
                             })

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
            return [n.ds_num_of_pre_synapses for n in self.neurons.values()]
        else:
            return [n.ds_num_of_post_synapses for n in self.neurons.values()]

    def get_cell_type_degree_distribution(self, cell_type: str, type_space: list[str], side: SynapseSide) -> dict:
        """
        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :param side: SynapseSide (pre, post)
        :return: Distribution based on the cell_type of pre- / post-neurons (based on the side)
        """
        degree_dist_per_type = {type_: [] for type_ in type_space}
        neuron_side = 'ds_num_of_pre_synapses' if side == SynapseSide.pre else 'ds_num_of_post_synapses'

        for n in tqdm(self.neurons.values()):
            degree_dist_per_type[getattr(n, cell_type)].append(getattr(n, neuron_side))

        return degree_dist_per_type


if __name__ == "__main__":
    connectome = Connectome()
