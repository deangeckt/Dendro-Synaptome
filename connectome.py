import pickle
import numpy as np
from tqdm import tqdm

from neuron import Neuron
from synapse import Synapse
from connectome_types import SynapseDirection, cell_types, CONNECTOME_BASE_PATH


class Connectome:
    NeuronsDict = dict[int, Neuron]

    def __init__(self):
        with open(CONNECTOME_BASE_PATH, 'rb') as f:
            self.neurons: Connectome.NeuronsDict = pickle.load(f)
            self.synapses = self._get_connectome_inter_synapses()

            print('Connectome:')
            print(f'\t#neurons: {len(self.neurons.keys())}')
            print(f'\t#synapses: {len(self.synapses)}')

    def _get_connectome_inter_synapses(self) -> list[Synapse]:
        """
        :return: a list of (inter) synapses connecting two neurons in the connectome
        """
        synapses = []
        conn_pre_synapses = 0
        conn_post_synapses = 0
        for neuron in tqdm(self.neurons.values()):
            synapses.extend([syn for syn in neuron.pre_synapses if syn.pre_pt_root_id in self.neurons])
            conn_pre_synapses += len(neuron.pre_synapses)
            conn_post_synapses += len(neuron.post_synapses)

        print(f'\t#pre  synapses in the connectome: {conn_pre_synapses}')
        print(f'\t#post synapses in the connectome: {conn_post_synapses}')

        for syn in tqdm(synapses):
            assert syn.dist_to_post_syn_soma != -1.0

        return synapses

    def get_cell_type_conn_matrix(self, cell_type: str, type_space: list[str]) -> np.ndarray:
        """
        :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
        :param type_space: list[str]: all possible types of cell_type
        :return: connectivity matrix of type cell_type
        """
        conn_matrix = np.zeros((len(type_space), len(type_space)), dtype=int)
        type_index = {t: i for i, t in enumerate(type_space)}

        for syn in self.synapses:
            pre_syn_neuron_type = getattr(self.neurons[syn.pre_pt_root_id], cell_type)
            post_syn_neuron_type = getattr(self.neurons[syn.post_pt_root_id], cell_type)
            conn_matrix[type_index[pre_syn_neuron_type], type_index[post_syn_neuron_type]] += 1

        return conn_matrix

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

        for syn in self.synapses:
            pre_syn_neuron_type = getattr(self.neurons[syn.pre_pt_root_id], cell_type)
            post_syn_neuron_type = getattr(self.neurons[syn.post_pt_root_id], cell_type)

            syn_attr_data = getattr(syn, syn_attr)
            syn_data = (syn.id_, syn_attr_data)

            if direction == SynapseDirection.output:
                synapse_attributes[pre_syn_neuron_type].append(syn_data)
            else:
                synapse_attributes[post_syn_neuron_type].append(syn_data)

        return synapse_attributes


if __name__ == "__main__":
    connectome = Connectome()
    d = connectome.get_cell_type_synapse_attr('cell_type',
                                              cell_types,
                                              SynapseDirection.output,
                                              'dist_to_post_syn_soma')
    print(d)
