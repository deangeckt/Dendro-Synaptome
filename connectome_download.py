import os
import pickle
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from tqdm import tqdm
import json

from connectome import Connectome
from connectome_offline_utils import calculate_synapse_dist_to_post_syn_soma
from neuron import Neuron
from synapse import Synapse
from connectome_types import ClfType, CONNECTOME_BASE_PATH, SKELETONS_DIR_PATH, NEURONS_PATH


def syn_table_to_synapses(df: pd.DataFrame) -> list[Synapse]:
    ids = df['id'].to_numpy()
    pre_ids = df['pre_pt_root_id'].to_numpy()
    post_ids = df['post_pt_root_id'].to_numpy()
    sizes = df['size'].to_numpy()
    center_positions = df['ctr_pt_position'].apply(np.array).tolist()
    return [Synapse(id_=ids[i], pre_pt_root_id=pre_ids[i], post_pt_root_id=post_ids[i], size=sizes[i],
                    center_position=center_positions[i], dist_to_post_syn_soma=-1.0) for i in range(len(df))
            ]


def download_neurons_dataset():
    client = CAVEclient('minnie65_public')

    cell_types = pd.read_csv('data/aibs_metamodel_celltypes_v661.csv')
    cell_types = cell_types[cell_types.classification_system != 'nonneuron']
    cell_types.set_index('root_id', inplace=True)

    m_types = pd.read_csv('data/aibs_metamodel_mtypes_v661_v2.csv')
    m_types = m_types[m_types.root_id != 0]
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
                            pre_synapses=syn_table_to_synapses(client.materialize.synapse_query(post_ids=cell_id)),
                            post_synapses=syn_table_to_synapses(client.materialize.synapse_query(pre_ids=cell_id)),
                            )
            with open(neuron_file_path, 'wb') as f:
                pickle.dump(neuron, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f'err in {cell_id}')
            print(e)


def combine_neurons_dataset():
    neurons: Connectome.NeuronsDict = {}
    for filename in os.listdir(NEURONS_PATH):
        file = os.path.join(NEURONS_PATH, filename)
        with open(file, 'rb') as f:
            neuron: Neuron = pickle.load(f)
            neurons[neuron.root_id] = neuron

    calculate_synapse_dist_to_post_syn_soma(neurons)

    with open(CONNECTOME_BASE_PATH, 'wb') as f:
        pickle.dump(neurons, f, protocol=pickle.HIGHEST_PROTOCOL)


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


if __name__ == "__main__":
    # combine_neurons_dataset()
    # download_neuron_skeletons()
    download_neurons_dataset()
