import pickle
from typing import TypedDict

import numpy as np
import pandas
import pandas as pd
from tqdm import tqdm

from connectome_types import CONNECTOME_BASE_PATH, CONNECTOME_TOY_PATH, ClfType, CONNECTOME_SYN_TABLE_PATH
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

    def get_neuron_table(self) -> pandas.DataFrame:
        neurons = self.neurons.values()
        root_id = [n.root_id for n in neurons]
        volume = [n.volume for n in neurons]
        clf_type = [n.clf_type for n in neurons]
        cell_type = [n.cell_type for n in neurons]
        mtype = [n.mtype for n in neurons]
        pre_synapses = [len(n.pre_synapses) for n in neurons]
        post_synapses = [n.num_of_post_synapses for n in neurons]

        # Dynamic properties
        pre_syn_weight = []
        ex_pre_syn_weight = []
        inh_pre_syn_weight = []
        for neuron in tqdm(neurons):
            pre_syn_weight.append(np.mean(np.array([syn.size for syn in neuron.pre_synapses])))
            ex_pre_syn_weight.append(np.mean(np.array([syn.size for syn in neuron.pre_synapses if self.neurons[
                syn.pre_pt_root_id].clf_type == ClfType.excitatory])))
            inh_pre_syn_weight.append(np.mean(np.array([syn.size for syn in neuron.pre_synapses if self.neurons[
                syn.pre_pt_root_id].clf_type == ClfType.inhibitory])))

        # for the whole dataset, not just the EM volume
        ds_num_of_pre_synapses = [n.ds_num_of_pre_synapses for n in neurons]
        ds_num_of_post_synapses = [n.ds_num_of_post_synapses for n in neurons]
        ds_pre_syn_mean_weight = [n.ds_pre_syn_mean_weight for n in neurons]
        ds_post_syn_mean_weight = [n.ds_post_syn_mean_weight for n in neurons]
        ds_pre_syn_sum_weight = [n.ds_pre_syn_sum_weight for n in neurons]
        ds_post_syn_sum_weight = [n.ds_post_syn_sum_weight for n in neurons]

        return pd.DataFrame({'root_id': root_id, 'volume': volume, 'clf_type': clf_type, 'cell_type': cell_type,
                             'mtype': mtype,
                             'ds_num_of_pre_synapses': ds_num_of_pre_synapses,
                             'ds_num_of_post_synapses': ds_num_of_post_synapses,
                             'ds_pre_syn_mean_weight': ds_pre_syn_mean_weight,
                             'ds_post_syn_mean_weight': ds_post_syn_mean_weight,
                             'ds_pre_syn_sum_weight': ds_pre_syn_sum_weight,
                             'ds_post_syn_sum_weight': ds_post_syn_sum_weight,
                             'num_of_pre_synapses': pre_synapses, 'num_of_post_synapses': post_synapses,
                             'pre_syn_weight': pre_syn_weight,
                             'ex_pre_syn_weight': ex_pre_syn_weight,
                             'inh_pre_syn_weight': inh_pre_syn_weight
                             })

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
        :return: connectivity matrix of type cell_type (sums)
        """
        conn_matrix = np.zeros((len(type_space), len(type_space)), dtype=int)
        type_index = {t: i for i, t in enumerate(type_space)}
        for syn in tqdm(self.synapses):
            pre_syn_neuron_type = getattr(self.neurons[syn.pre_pt_root_id], cell_type)
            post_syn_neuron_type = getattr(self.neurons[syn.post_pt_root_id], cell_type)
            conn_matrix[type_index[post_syn_neuron_type], type_index[pre_syn_neuron_type]] += 1
        return conn_matrix

    def __get_neuron_counts(self, cell_type: str, type_space: list[str]) -> np.ndarray:
        """
        Count the number of neurons for each type in type_space.

        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :return: array of neuron counts for each type
        """
        counts = np.zeros(len(type_space), dtype=int)
        type_index = {t: i for i, t in enumerate(type_space)}

        for neuron in self.neurons.values():
            neuron_type = getattr(neuron, cell_type)
            counts[type_index[neuron_type]] += 1

        return counts

    def get_average_incoming_conn_matrix(self, cell_type: str, type_space: list[str]) -> np.ndarray:
        """
        Calculate the average incoming connectivity matrix.

        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :return: average incoming connectivity matrix
        """
        conn_matrix = self.get_cell_type_conn_matrix(cell_type, type_space)
        neuron_counts = self.__get_neuron_counts(cell_type, type_space)

        # Divide each column by the number of neurons of that type
        avg_incoming_matrix = conn_matrix / neuron_counts[np.newaxis, :]

        return avg_incoming_matrix

    def get_average_outgoing_conn_matrix(self, cell_type: str, type_space: list[str]) -> np.ndarray:
        """
        Calculate the average outgoing connectivity matrix.

        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :return: average outgoing connectivity matrix
        """
        conn_matrix = self.get_cell_type_conn_matrix(cell_type, type_space)
        neuron_counts = self.__get_neuron_counts(cell_type, type_space)

        # Divide each row by the number of neurons of that type
        avg_outgoing_matrix = conn_matrix / neuron_counts[:, np.newaxis]

        return avg_outgoing_matrix


if __name__ == "__main__":
    connectome = Connectome()
