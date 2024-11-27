import pickle
import numpy as np

from neuron import Neuron
from synapse import Synapse
from connectome_types import SynapseDirection, cell_types, CONNECTOME_BASE_PATH


class Connectome:
    ConnectomeType = dict[int, Neuron]

    def __init__(self):
        with open(CONNECTOME_BASE_PATH, 'rb') as f:
            self.connectome: Connectome.ConnectomeType = pickle.load(f)
            self.synapses = self._get_connectome_inter_synapses()

            print('Connectome:')
            print(f'\t#neurons: {len(self.connectome.keys())}')
            print(f'\t#synapses: {len(self.synapses)}')

    def _get_connectome_inter_synapses(self) -> list[Synapse]:
        """
        :return: a list of synapses connecting two neurons in the connectome
        """
        synapses = []
        for neuron in self.connectome.values():
            synapses.extend([syn for syn in neuron.post_synapses if syn.post_pt_root_id in self.connectome])
        return synapses

    def calculate_cell_type_conn_matrix(self, cell_type: str, type_space: list[str]) -> np.ndarray:
        """
        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :return: connectivity matrix of type cell_type
        """

        conn_matrix = np.zeros((len(type_space), len(type_space)), dtype=int)
        type_index = {t: i for i, t in enumerate(type_space)}

        for syn in self.synapses:
            pre_syn_neuron_type = getattr(self.connectome[syn.pre_pt_root_id], cell_type)
            post_syn_neuron_type = getattr(self.connectome[syn.post_pt_root_id], cell_type)
            conn_matrix[type_index[pre_syn_neuron_type], type_index[post_syn_neuron_type]] += 1

        return conn_matrix

    def calculate_cell_type_synapse_attr(self,
                                         cell_type: str,
                                         type_space: list[str],
                                         direction: SynapseDirection,
                                         syn_attr: str,
                                         ) -> dict:
        """
        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :param direction: SynapseDirection (input, output)
        :param syn_attr: str: (size)
        :return: data represented in a dict, where each key is a cell_type, and the value are list of all
        aggregated values of the given calculated attribute in a tuple format: (syn_id, attr).
        """
        synapse_attributes = {type_: [] for type_ in type_space}

        for syn in self.synapses:
            pre_syn_neuron_type = getattr(self.connectome[syn.pre_pt_root_id], cell_type)
            post_syn_neuron_type = getattr(self.connectome[syn.post_pt_root_id], cell_type)

            syn_attr_data = getattr(syn, syn_attr)
            syn_data = (syn.id_, syn_attr_data)

            if direction == SynapseDirection.output:
                synapse_attributes[pre_syn_neuron_type].append(syn_data)
            else:
                synapse_attributes[post_syn_neuron_type].append(syn_data)

        return synapse_attributes

    def calculate_cell_type_synapse_dist_to_soma(self, cell_type: str, type_space: list[str]) -> dict:
        """
        Calculate the distances of synapses to a post-synaptic neuron soma, aggregated according to cell type
        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :return: data represented in a dict, where each key is a cell_type, and the value are list of all
        distances to soma, tuple format: (syn_id, dist).
        """
        distances = {type_: [] for type_ in type_space}

        for neuron in self.connectome.values():
            sk = neuron.load_skeleton()
            if sk is None:
                continue

            all_syn_xyz = [syn.center_position * np.array([4, 4, 40]) for syn in neuron.pre_synapses]
            syn_ds, sk_syn_inds = sk.kdtree.query(all_syn_xyz)
            distances_to_soma = [sk.distance_to_root[s] for s in sk_syn_inds]

            for syn_idx, syn in enumerate(neuron.pre_synapses):
                if syn.pre_pt_root_id not in self.connectome:
                    continue
                pre_syn_neuron_type = getattr(self.connectome[syn.pre_pt_root_id], cell_type)
                syn_dist = distances_to_soma[syn_idx]
                syn_data = (syn.id_, syn_dist)
                distances[pre_syn_neuron_type].append(syn_data)

        return distances


if __name__ == "__main__":
    connectome = Connectome()
    d = connectome.calculate_cell_type_synapse_attr('cell_type',
                                                    cell_types,
                                                    SynapseDirection.output,
                                                    'size')
    print(d)
