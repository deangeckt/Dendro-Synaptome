import os
import pickle
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from tqdm import tqdm
import json

from connectome import NeuronsDict, Connectome
from connectome_offline_utils import validate_neurons_files_and_skeletons, \
    calculate_synapse_depth, calculate_synapse_dist_to_post_syn_soma, calculate_synapse_dist_to_pre_syn_soma
from neuron import Neuron
from synapse import Synapse
from connectome_types import ClfType, SKELETONS_DIR_PATH, NEURONS_PATH, \
    EM_NEURONS_PATH, CONNECTOME_SYN_TABLE_PATH, CONNECTOME_NEURON_TABLE_PATH


def __syn_table_to_synapses(df: pd.DataFrame) -> list[Synapse]:
    ids = df['id'].to_numpy()
    pre_ids = df['pre_pt_root_id'].to_numpy()
    post_ids = df['post_pt_root_id'].to_numpy()
    sizes = df['size'].to_numpy()
    center_positions = df['ctr_pt_position'].apply(np.array).tolist()
    return [Synapse(id_=ids[i], pre_pt_root_id=pre_ids[i], post_pt_root_id=post_ids[i], size=sizes[i],
                    center_position=center_positions[i]) for i in range(len(df))
            ]


def download_neurons_dataset():
    client = CAVEclient('minnie65_public')
    cell_types = pd.read_csv('data/aibs_metamodel_celltypes_v661.csv')
    cell_types = cell_types[cell_types.classification_system != 'nonneuron']

    m_types = pd.read_csv('data/aibs_metamodel_mtypes_v661_v2.csv')
    m_types = m_types[m_types.root_id != 0]

    duplications = [864691136101343093, 864691135495542672, 864691136990457749, 864691135952250403]
    mask = ~((cell_types['root_id'].isin(duplications)) & (cell_types.duplicated('root_id', keep='last')))
    cell_types = cell_types[mask]

    mask = ~((m_types['root_id'].isin(duplications)) & (m_types.duplicated('root_id', keep='last')))
    m_types = m_types[mask]

    cell_types.set_index('root_id', inplace=True)
    m_types.set_index('root_id', inplace=True)

    os.makedirs(NEURONS_PATH, exist_ok=True)
    for cell_id in tqdm(m_types.index):
        neuron_file_path = f'{NEURONS_PATH}/{cell_id}.pkl'
        if os.path.exists(neuron_file_path):
            continue
        try:
            clf_type = ClfType.excitatory if (cell_types.at[cell_id, 'classification_system']
                                              == 'excitatory_neuron') else ClfType.inhibitory
            neuron = Neuron(root_id=cell_id,
                            clf_type=clf_type,
                            cell_type=cell_types.at[cell_id, 'cell_type'],
                            mtype=m_types.at[cell_id, 'cell_type'],
                            position=cell_types.loc[
                                cell_id, ['pt_position_x', 'pt_position_y', 'pt_position_z']].to_numpy(),
                            volume=cell_types.at[cell_id, 'volume'],
                            pre_synapses=__syn_table_to_synapses(client.materialize.synapse_query(post_ids=cell_id)),
                            post_synapses=__syn_table_to_synapses(client.materialize.synapse_query(pre_ids=cell_id)),
                            )
            with open(neuron_file_path, 'wb') as f:
                pickle.dump(neuron, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f'err in {cell_id}')
            print(e)


def download_neuron_skeletons():
    client = CAVEclient('minnie65_public')
    m_types = pd.read_csv('data/aibs_metamodel_mtypes_v661_v2.csv')
    m_types = m_types[m_types.root_id != 0]
    m_types.set_index('root_id', inplace=True)
    os.makedirs(SKELETONS_DIR_PATH, exist_ok=True)

    for cell_id in tqdm(m_types.index):
        sk_file_path = f'{SKELETONS_DIR_PATH}/{cell_id}.json'
        if os.path.exists(sk_file_path):
            continue
        try:
            sk_dict = client.skeleton.get_skeleton(cell_id, output_format='json')
            with open(sk_file_path, 'w') as f:
                json.dump(sk_dict, f)
        except Exception as e:
            print(f'err in {cell_id}')
            print(e)


def create_neurons_em_dataset():
    """
    Save to EM_NEURONS_PATH all the neurons, left only with their pre-synapses
    that connects two neurons from EM volume
    :return:
    """
    os.makedirs(EM_NEURONS_PATH, exist_ok=True)
    neurons_ids = {int(f.split('.')[0]) for f in os.listdir(NEURONS_PATH)}
    dataset_pre_synapses = 0
    dataset_post_synapses = 0

    for filename in tqdm(os.listdir(NEURONS_PATH)):
        with open(os.path.join(NEURONS_PATH, filename), 'rb') as f:
            neuron: Neuron = pickle.load(f)
            dataset_pre_synapses += len(neuron.pre_synapses)
            dataset_post_synapses += len(neuron.post_synapses)

            pre_synapses = [syn for syn in neuron.pre_synapses if syn.pre_pt_root_id in neurons_ids]
            post_synapses = [syn for syn in neuron.post_synapses if syn.post_pt_root_id in neurons_ids]

            if not pre_synapses and not post_synapses:
                continue

            neuron.ds_pre_syn_mean_weight = np.mean(np.array([syn.size for syn in neuron.pre_synapses]))
            neuron.ds_post_syn_mean_weight = np.mean(np.array([syn.size for syn in neuron.post_synapses]))

            neuron.ds_pre_syn_std_weight = np.std(np.array([syn.size for syn in neuron.pre_synapses]))
            neuron.ds_post_syn_std_weight = np.std(np.array([syn.size for syn in neuron.post_synapses]))

            neuron.ds_pre_syn_sum_weight = np.sum(np.array([syn.size for syn in neuron.pre_synapses]))
            neuron.ds_post_syn_sum_weight = np.sum(np.array([syn.size for syn in neuron.post_synapses]))

            neuron.ds_num_of_post_synapses = len(neuron.post_synapses)
            neuron.ds_num_of_pre_synapses = len(neuron.pre_synapses)

            # override with inter-synapses
            neuron.post_synapses = post_synapses
            neuron.pre_synapses = pre_synapses

        with open(os.path.join(EM_NEURONS_PATH, filename), 'wb') as f:
            pickle.dump(neuron, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'#pre  synapses in the dataset: {dataset_pre_synapses}')
    print(f'#post synapses in the dataset: {dataset_post_synapses}')


def override_neurons_em_dataset_attributes():
    for filename in tqdm(os.listdir(EM_NEURONS_PATH)):
        neuron_file_path = os.path.join(EM_NEURONS_PATH, filename)

        with open(neuron_file_path, 'rb') as f:
            neuron: Neuron = pickle.load(f)

            if not neuron.pre_synapses:
                continue
            syn = neuron.pre_synapses[0]
            if not hasattr(syn, 'dist_to_post_syn_soma') or syn.dist_to_post_syn_soma == -1.0:
                calculate_synapse_dist_to_post_syn_soma(neuron)
            if not hasattr(syn, 'dist_to_pre_syn_soma') or syn.dist_to_pre_syn_soma == -1.0:
                calculate_synapse_dist_to_pre_syn_soma(neuron)
            if not hasattr(syn, 'depth') or syn.depth == -1.0:
                calculate_synapse_depth(neuron)

        with open(os.path.join(EM_NEURONS_PATH, filename), 'wb') as f:
            pickle.dump(neuron, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_neurons_full_dict() -> NeuronsDict:
    neurons: NeuronsDict = {}
    for filename in tqdm(os.listdir(EM_NEURONS_PATH)):
        with open(os.path.join(EM_NEURONS_PATH, filename), 'rb') as f:
            neuron: Neuron = pickle.load(f)
            neurons[neuron.root_id] = neuron

    return neurons


def combine_neurons_dataset():
    validate_neurons_files_and_skeletons()
    neurons: NeuronsDict = load_neurons_full_dict()

    synapses: list[Synapse] = []
    for n in tqdm(neurons.values()):
        synapses.extend(n.pre_synapses)

    conn = Connectome(neurons=neurons, synapses=synapses, from_disk=False)
    conn.synapses.to_csv(CONNECTOME_SYN_TABLE_PATH)
    conn.neurons.to_csv(CONNECTOME_NEURON_TABLE_PATH)


if __name__ == "__main__":
    # create_neurons_em_dataset()

    # override_neurons_em_dataset_attributes()
    combine_neurons_dataset()

    # download_neuron_skeletons()
    # download_neurons_dataset()
