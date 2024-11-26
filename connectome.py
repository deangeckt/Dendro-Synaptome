import os
import pickle
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from tqdm import tqdm
import json

from neuron import Neuron
from synapse import Synapse
from connectome_types import ClfType, SynapseDirection

CONNECTOME_BASE_PATH = 'data/connectome_base.pkl'
SKELETONS_DIR_PATH = 'data/skeletons'


def syn_table_to_synapses(df: pd.DataFrame) -> list[Synapse]:
    ids = df['id'].to_numpy()
    pre_ids = df['pre_pt_root_id'].to_numpy()
    post_ids = df['post_pt_root_id'].to_numpy()
    sizes = df['size'].to_numpy()
    center_positions = df['ctr_pt_position'].apply(np.array).tolist()
    return [Synapse(id_=ids[i], pre_pt_root_id=pre_ids[i], post_pt_root_id=post_ids[i], size=sizes[i],
                    center_position=center_positions[i]) for i in range(len(df))
            ]


def download_dataset():
    # TODO: batch
    client = CAVEclient('minnie65_public')
    cell_types = pd.read_csv('data/aibs_metamodel_celltypes_v661.csv')
    m_types = pd.read_csv('data/aibs_metamodel_mtypes_v661_v2.csv')
    cell_types.set_index('root_id', inplace=True)
    m_types.set_index('root_id', inplace=True)

    connectome = {}
    for cell_id in tqdm(m_types.index):
        if cell_id == 0:
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
            connectome[cell_id] = neuron
        except Exception as e:
            print(f'err in {cell_id}')
            print(e)

    with open(CONNECTOME_BASE_PATH, 'wb') as f:
        pickle.dump(connectome, f, protocol=pickle.HIGHEST_PROTOCOL)


def download_neuron_skeletons():
    client = CAVEclient('minnie65_public')
    m_types = pd.read_csv('data/aibs_metamodel_mtypes_v661_v2.csv')
    m_types.set_index('root_id', inplace=True)
    os.makedirs(SKELETONS_DIR_PATH, exist_ok=True)

    for cell_id in tqdm(m_types.index):
        if cell_id == 0:
            continue
        try:
            sk_dict = client.skeleton.get_skeleton(cell_id, output_format='json')

            with open(f'{SKELETONS_DIR_PATH}/{cell_id}.json', 'w') as f:
                json.dump(sk_dict, f)
        except Exception as e:
            print(f'err in {cell_id}')
            print(e)


def load_local_dataset() -> dict:
    with open(CONNECTOME_BASE_PATH, 'rb') as f:
        return pickle.load(f)


def calculate_cell_type_conn_matrix(cell_type: str,
                                    type_space: list[str],
                                    direction: SynapseDirection) -> np.ndarray:
    """
    :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
    :param type_space: list[str]: all possible types of cell_type
    :param direction: SynapseDirection (input, output)
    :return: connectivity matrix
    """
    connectome = load_local_dataset()
    matrix = np.zeros((len(type_space), len(type_space)), dtype=int)
    type_index = {t: i for i, t in enumerate(type_space)}

    for neuron in connectome.values():
        src_type = getattr(neuron, cell_type)
        synapses: list[Synapse] = neuron.post_synapses if direction == SynapseDirection.output else neuron.pre_synapses
        for syn in synapses:
            target_neuron_id = syn.post_pt_root_id if direction == SynapseDirection.output else syn.pre_pt_root_id
            if target_neuron_id not in connectome:
                continue
            connected_neuron: Neuron = connectome[target_neuron_id]
            target_type = getattr(connected_neuron, cell_type)

            # keep it pre-synaptic X post-synaptic
            if direction == SynapseDirection.output:
                matrix[type_index[src_type], type_index[target_type]] += 1
            else:
                matrix[type_index[target_type], type_index[src_type]] += 1

    return matrix


def aggregate_():
    pass


if __name__ == "__main__":
    # download_neuron_skeletons()
    download_dataset()
    # calculate_cell_type_conn_matrix('clf_type', ['E', 'I'], SynapseDirection.input)
