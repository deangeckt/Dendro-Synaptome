import os
import pickle

import meshparty.skeleton_io
from meshparty import skeleton
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from tqdm import tqdm
import json

from neuron import Neuron
from synapse import Synapse
from connectome_types import ClfType, SynapseDirection, cell_types

DATA_BASE_PATH = 'data/'
CONNECTOME_BASE_PATH = 'data/connectome_base_100.pkl'
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
    os.makedirs(DATA_BASE_PATH, exist_ok=True)

    client = CAVEclient('minnie65_public')
    cell_types = pd.read_csv('data/aibs_metamodel_celltypes_v661.csv')
    m_types = pd.read_csv('data/aibs_metamodel_mtypes_v661_v2.csv')
    cell_types.set_index('root_id', inplace=True)
    m_types.set_index('root_id', inplace=True)

    batch_size = 10
    connectome = {}

    for start in tqdm(range(0, len(m_types), batch_size)):
        end = min(start + batch_size, len(m_types))
        batch_ids = m_types.index[start:end]

        batch_ids = batch_ids[batch_ids != 0]
        if len(batch_ids) == 0:
            continue

        try:
            classification_systems = cell_types.loc[batch_ids, 'classification_system'].values
            clf_types = [ClfType.excitatory if s == 'excitatory_neuron' else ClfType.inhibitory for s in
                         classification_systems]

            mtype_values = m_types.loc[batch_ids, 'cell_type']
            cell_type_values = cell_types.loc[batch_ids, 'cell_type']
            positions = cell_types.loc[batch_ids, ['pt_position_x', 'pt_position_y', 'pt_position_z']].to_numpy()
            volumes = cell_types.loc[batch_ids, 'volume']

            # Batch fashion
            pre_synapses = syn_table_to_synapses(client.materialize.synapse_query(post_ids=list(batch_ids.values)))
            post_synapses = syn_table_to_synapses(client.materialize.synapse_query(pre_ids=list(batch_ids.values)))

            for i, cell_id in enumerate(batch_ids):
                cell_post_synapses = [syn for syn in post_synapses if syn.pre_pt_root_id == cell_id]
                cell_pre_synapses = [syn for syn in pre_synapses if syn.post_pt_root_id == cell_id]

                neuron = Neuron(root_id=cell_id,
                                clf_type=clf_types[i],
                                cell_type=cell_type_values.iloc[i],
                                mtype=mtype_values.iloc[i],
                                position=positions[i],
                                volume=volumes.iloc[i],
                                pre_synapses=cell_pre_synapses,
                                post_synapses=cell_post_synapses)

                connectome[cell_id] = neuron

        except Exception as e:
            print(f'Error in chunk {start}. cell_ids {batch_ids}')
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
    :return: connectivity matrix, i.e.: number of out/in going connections of type cell_type
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


def calculate_cell_type_synapse_attr(cell_type: str,
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
    connectome = load_local_dataset()
    data = {type_: [] for type_ in type_space}

    for neuron in connectome.values():
        src_type = getattr(neuron, cell_type)
        synapses: list[Synapse] = neuron.post_synapses if direction == SynapseDirection.output else neuron.pre_synapses
        for syn in synapses:
            target_neuron_id = syn.post_pt_root_id if direction == SynapseDirection.output else syn.pre_pt_root_id
            if target_neuron_id not in connectome:
                continue
            connected_neuron: Neuron = connectome[target_neuron_id]
            target_type = getattr(connected_neuron, cell_type)
            syn_attr_data = getattr(syn, syn_attr)
            syn_data = (syn.id_, syn_attr_data)

            data_list = data[src_type] if direction == SynapseDirection.output else data[target_type]
            data_list.append(syn_data)

    return data


def calculate_cell_type_synapse_dist_to_soma(cell_type: str,
                                             type_space: list[str],
                                             direction: SynapseDirection,
                                             ) -> dict:
    """
    :param cell_type: str: (mtype, cell_type, clf_type) which are attributes of neuron class
    :param type_space: list[str]: all possible types of cell_type
    :param direction: SynapseDirection (input, output)
    :return: data represented in a dict, where each key is a cell_type, and the value are list of all
    distances to soma, tuple format: (syn_id, dist).
    """
    connectome = load_local_dataset()
    data = {type_: [] for type_ in type_space}

    for neuron in connectome.values():
        cell_id = neuron.root_id
        sk_file_path = f'{SKELETONS_DIR_PATH}/{cell_id}.json'
        if not os.path.exists(sk_file_path) or os.path.getsize(sk_file_path) == 0:
            continue

        with open(sk_file_path) as f:
            sk_dict = json.load(f)

            sk = meshparty.skeleton.Skeleton(
                vertices=np.array(sk_dict['vertices']),
                edges=np.array(sk_dict['edges']),
                mesh_to_skel_map=sk_dict['mesh_to_skel_map'],
                vertex_properties=sk_dict['vertex_properties'],
                root=sk_dict['root'],
                meta=sk_dict['meta'],
            )

            src_type = getattr(neuron, cell_type)
            synapses: list[Synapse] = neuron.post_synapses if direction == SynapseDirection.output else neuron.pre_synapses
            all_syn_xyz = [syn.center_position * np.array([4,4,40]) for syn in synapses]
            syn_ds, sk_syn_inds = sk.kdtree.query(all_syn_xyz)
            distances_to_soma = [sk.distance_to_root[s] for s in sk_syn_inds]

            for syn_idx, syn in enumerate(synapses):
                target_neuron_id = syn.post_pt_root_id if direction == SynapseDirection.output else syn.pre_pt_root_id
                if target_neuron_id not in connectome:
                    continue
                connected_neuron: Neuron = connectome[target_neuron_id]
                target_type = getattr(connected_neuron, cell_type)
                syn_dist = distances_to_soma[syn_idx]
                syn_data = (syn.id_, syn_dist)

                data_list = data[src_type] if direction == SynapseDirection.output else data[target_type]
                data_list.append(syn_data)

    return data


if __name__ == "__main__":
    # download_neuron_skeletons()
    # download_dataset()
    # calculate_cell_type_conn_matrix('clf_type', ['E', 'I'], SynapseDirection.input)
    # d = calculate_cell_type_synapse_attr('cell_type', cell_types, SynapseDirection.input, 'size')
    d = calculate_cell_type_synapse_dist_to_soma('cell_type', cell_types, SynapseDirection.input)
    print(d)

