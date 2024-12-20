import numpy as np
import pandas
import pandas as pd
from tqdm import tqdm

from connectome_types import ClfType, CONNECTOME_SYN_TABLE_PATH, CONNECTOME_NEURON_TABLE_PATH
from neuron import Neuron
from synapse import Synapse

NeuronsDict = dict[int, Neuron]


class Connectome:
    """
    This Connectome represents a partial part of the whole MICrONS Dataset, It contains
    all neurons, but the synapses are those that each endpoint (i.e., the post/pre-synaptic neuron) belongs
    to the dataset, in other words, synapses coming from (pre-synaptic) neurons outside the EM volume are excluded,
    resulting in ~13M synapses instead of ~300M+.
    """

    def __init__(self,
                 neurons: NeuronsDict = None,
                 synapses: list[Synapse] = None,
                 from_disk=False):

        if from_disk:
            self.neurons = pd.read_csv(CONNECTOME_NEURON_TABLE_PATH)
            self.synapses = pd.read_csv(CONNECTOME_SYN_TABLE_PATH)
        else:
            self.neurons = self.get_neuron_table(neurons)
            self.synapses = self.get_synapses_table(synapses, neurons)

        print('Connectome:')
        print(f'\t#neurons: {len(self.neurons)}')
        print(f'\t#synapses: {len(self.synapses)}')
        self.neuron_lookup = self.neurons.set_index('root_id').to_dict(orient='index')

    @staticmethod
    def get_neuron_table(neurons_dict: NeuronsDict) -> pandas.DataFrame:
        neurons = neurons_dict.values()
        root_id = [n.root_id for n in neurons]
        volume = [n.volume for n in neurons]
        clf_type = [n.clf_type for n in neurons]
        cell_type = [n.cell_type for n in neurons]
        mtype = [n.mtype for n in neurons]

        pre_synapses = [len(n.pre_synapses) for n in neurons]
        post_synapses = [len(n.post_synapses) for n in neurons]

        ex_pre_synapses = [len([syn for syn in n.pre_synapses if neurons_dict[
            syn.pre_pt_root_id].clf_type == ClfType.excitatory]) for n in neurons]
        inh_pre_synapses = [len([syn for syn in n.pre_synapses if neurons_dict[
            syn.pre_pt_root_id].clf_type == ClfType.inhibitory]) for n in neurons]

        # Dynamic (incoming) properties
        pre_syn_mean_weight = []
        pre_syn_sum_weight = []
        pre_syn_std_weight = []

        ex_pre_syn_mean_weight = []
        ex_pre_syn_sum_weight = []
        ex_pre_syn_std_weight = []

        inh_pre_syn_mean_weight = []
        inh_pre_syn_sum_weight = []
        inh_pre_syn_std_weight = []

        for neuron in tqdm(neurons):
            weights = np.array([syn.size for syn in neuron.pre_synapses])
            pre_syn_mean_weight.append(np.mean(weights))
            pre_syn_sum_weight.append(np.sum(weights))
            pre_syn_std_weight.append(np.std(weights))

            ex_weights = np.array([syn.size for syn in neuron.pre_synapses if neurons_dict[
                syn.pre_pt_root_id].clf_type == ClfType.excitatory])
            ex_pre_syn_mean_weight.append(np.mean(ex_weights))
            ex_pre_syn_sum_weight.append(np.sum(ex_weights))
            ex_pre_syn_std_weight.append(np.std(ex_weights))

            inh_weights = np.array([syn.size for syn in neuron.pre_synapses if neurons_dict[
                syn.pre_pt_root_id].clf_type == ClfType.inhibitory])
            inh_pre_syn_mean_weight.append(np.mean(inh_weights))
            inh_pre_syn_sum_weight.append(np.sum(inh_weights))
            inh_pre_syn_std_weight.append(np.std(inh_weights))

        # for the whole dataset, not just the EM volume
        ds_num_of_incoming_synapses = [n.ds_num_of_incoming_synapses for n in neurons]
        ds_num_of_outgoing_synapses = [n.ds_num_of_outgoing_synapses for n in neurons]
        ds_incoming_syn_mean_weight = [n.ds_incoming_syn_mean_weight for n in neurons]
        ds_outgoing_syn_mean_weight = [n.ds_outgoing_syn_mean_weight for n in neurons]
        ds_incoming_syn_std_weight = [n.ds_incoming_syn_std_weight for n in neurons]
        ds_outgoing_syn_std_weight = [n.ds_outgoing_syn_std_weight for n in neurons]
        ds_incoming_syn_sum_weight = [n.ds_incoming_syn_sum_weight for n in neurons]
        ds_outgoing_syn_sum_weight = [n.ds_outgoing_syn_sum_weight for n in neurons]

        return pd.DataFrame({'root_id': root_id, 'volume': volume, 'clf_type': clf_type, 'cell_type': cell_type,
                             'mtype': mtype,
                             'ds_num_of_incoming_synapses': ds_num_of_incoming_synapses,
                             'ds_num_of_outgoing_synapses': ds_num_of_outgoing_synapses,
                             'ds_incoming_syn_mean_weight': ds_incoming_syn_mean_weight,
                             'ds_outgoing_syn_mean_weight': ds_outgoing_syn_mean_weight,
                             'ds_incoming_syn_sum_weight': ds_incoming_syn_sum_weight,
                             'ds_outgoing_syn_sum_weight': ds_outgoing_syn_sum_weight,
                             'ds_incoming_syn_std_weight': ds_incoming_syn_std_weight,
                             'ds_outgoing_syn_std_weight': ds_outgoing_syn_std_weight,
                             'num_of_incoming_synapses': pre_synapses,
                             'num_of_outgoing_synapses': post_synapses,
                             'num_of_ex_incoming_synapses': ex_pre_synapses,
                             'num_of_inh_incoming_synapses': inh_pre_synapses,
                             'incoming_syn_mean_weight': pre_syn_mean_weight,
                             'incoming_syn_sum_weight': pre_syn_sum_weight,
                             'incoming_syn_std_weight': pre_syn_std_weight,
                             'ex_incoming_syn_mean_weight': ex_pre_syn_mean_weight,
                             'ex_incoming_syn_sum_weight': ex_pre_syn_sum_weight,
                             'ex_incoming_syn_std_weight': ex_pre_syn_std_weight,
                             'inh_incoming_syn_mean_weight': inh_pre_syn_mean_weight,
                             'inh_incoming_syn_sum_weight': inh_pre_syn_sum_weight,
                             'inh_incoming_syn_std_weight': inh_pre_syn_std_weight
                             })

    @staticmethod
    def get_synapses_table(synapses: list[Synapse], neurons: NeuronsDict) -> pandas.DataFrame:
        synapses_size = []
        synapses_pos = []
        synapses_depth_in_post = []
        synapses_depth_in_pre = []
        syn_id = []

        dist_to_post = []
        dist_to_pre = []

        # pre-synaptic neuron data
        pre_ids = []
        pre_clf_type = []
        pre_cell_type_type = []
        pre_mtype_type = []

        # post-synaptic neuron data
        post_ids = []
        post_clf_type = []
        post_cell_type = []
        post_mtype_type = []

        for syn in tqdm(synapses):
            syn_id.append(syn.id_)
            synapses_size.append(syn.size)
            synapses_pos.append(syn.center_position * np.array([4, 4, 40]))

            depth_in_post_syn_tree = syn.depth_in_post_syn_tree if hasattr(syn, 'depth_in_post_syn_tree') else -1.0
            depth_in_pre_syn_tree = syn.depth_in_pre_syn_tree if hasattr(syn, 'depth_in_pre_syn_tree') else -1.0
            synapses_depth_in_post.append(depth_in_post_syn_tree)
            synapses_depth_in_pre.append(depth_in_pre_syn_tree)

            dist_to_post_soma = syn.dist_to_post_syn_soma if hasattr(syn, 'dist_to_post_syn_soma') else -1.0
            dist_to_pre_soma = syn.dist_to_pre_syn_soma if hasattr(syn, 'dist_to_pre_syn_soma') else -1.0
            dist_to_post.append(dist_to_post_soma / 1000)
            dist_to_pre.append(dist_to_pre_soma / 1000)

            pre_syn_neuron = neurons[syn.pre_pt_root_id]
            pre_ids.append(syn.pre_pt_root_id)
            pre_clf_type.append(pre_syn_neuron.clf_type)
            pre_cell_type_type.append(pre_syn_neuron.cell_type)
            pre_mtype_type.append(pre_syn_neuron.mtype)

            post_syn_neuron = neurons[syn.post_pt_root_id]
            post_ids.append(syn.post_pt_root_id)
            post_clf_type.append(post_syn_neuron.clf_type)
            post_cell_type.append(post_syn_neuron.cell_type)
            post_mtype_type.append(post_syn_neuron.mtype)

        return pd.DataFrame({'id_': syn_id, 'pre_id': pre_ids, 'post_id': post_ids,
                             'dist_to_post_syn_soma': dist_to_post, 'dist_to_pre_syn_soma': dist_to_pre,
                             'size': synapses_size, 'center_position': synapses_pos,
                             'synapses_depth_in_post': synapses_depth_in_post,
                             'synapses_depth_in_pre': synapses_depth_in_pre,
                             'pre_clf_type': pre_clf_type, 'pre_cell_type': pre_cell_type_type,
                             'pre_m_type': pre_mtype_type, 'post_clf_type': post_clf_type,
                             'post_cell_type': post_cell_type, 'post_mtype_type': post_mtype_type,
                             })

    def get_neuron_conn_matrix(self, type_: ClfType) -> np.ndarray:
        if type_ == ClfType.both:
            filtered_neurons_set = set(self.neurons.root_id)
            filtered_neurons_list = list(self.neurons.root_id)
        else:
            filtered_neurons_set = set(self.neurons[self.neurons.clf_type == type_].root_id)
            filtered_neurons_list = list(self.neurons[self.neurons.clf_type == type_].root_id)

        conn_matrix = np.zeros((len(filtered_neurons_list), len(filtered_neurons_list)), dtype=np.int64)
        synapses = list(zip(list(self.synapses.pre_id), list(self.synapses.post_id)))

        for pre_syn, post_syn in tqdm(synapses):
            if pre_syn not in filtered_neurons_set or post_syn not in filtered_neurons_set:
                continue
            conn_matrix[filtered_neurons_list.index(post_syn), filtered_neurons_list.index(pre_syn)] += 1

        return conn_matrix

    def get_cell_type_conn_matrix(self, cell_type: str, type_space: list[str]) -> np.ndarray:
        """
        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :return: connectivity matrix of type cell_type (sums)
        """
        conn_matrix = np.zeros((len(type_space), len(type_space)), dtype=int)
        type_index = {t: i for i, t in enumerate(type_space)}
        synapses = list(zip(list(self.synapses.pre_id), list(self.synapses.post_id)))

        for pre_syn, post_syn in tqdm(synapses):
            pre_syn_neuron = self.neuron_lookup[pre_syn]
            post_syn_neuron = self.neuron_lookup[post_syn]
            pre_syn_neuron_type = pre_syn_neuron[cell_type]
            post_syn_neuron_type = post_syn_neuron[cell_type]
            conn_matrix[type_index[post_syn_neuron_type], type_index[pre_syn_neuron_type]] += 1

        return conn_matrix

    def get_cell_type_conn_matrix_of_syn_attr(self, cell_type: str, type_space: list[str],
                                              attr: str) -> np.ndarray:
        """
        :param attr:  str: attribute of syn class to aggregate by
        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :return: connectivity matrix of type cell_type (sums) aggregated according to the given neuron attribute
        """
        conn_matrix = np.zeros((len(type_space), len(type_space)), dtype=np.int64)
        type_index = {t: i for i, t in enumerate(type_space)}
        synapses = list(zip(self.synapses.pre_id, self.synapses.post_id, self.synapses[attr]))

        for pre_syn, post_syn, attr in tqdm(synapses):
            pre_syn_neuron = self.neuron_lookup[pre_syn]
            post_syn_neuron = self.neuron_lookup[post_syn]
            pre_syn_neuron_type = pre_syn_neuron[cell_type]
            post_syn_neuron_type = post_syn_neuron[cell_type]
            conn_matrix[type_index[post_syn_neuron_type], type_index[pre_syn_neuron_type]] += attr

        return conn_matrix


if __name__ == "__main__":
    pass