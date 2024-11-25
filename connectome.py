import pickle
import numpy as np
import pandas as pd
from caveclient import CAVEclient
from tqdm import tqdm

from neuron import Neuron, ClfType
from synapse import Synapse

CONNECTOME_BASE_PATH = 'data/connectome_base.pkl'


def syn_table_to_synapses(df: pd.DataFrame) -> list[Synapse]:
    ids = df['id'].to_numpy()
    pre_ids = df['pre_pt_root_id'].to_numpy()
    post_ids = df['post_pt_root_id'].to_numpy()
    sizes = df['size'].to_numpy()
    center_positions = df['ctr_pt_position'].apply(np.array).tolist()
    return [Synapse(id_=ids[i], pre_pt_root_id=pre_ids[i], post_pt_root_id=post_ids[i], size=sizes[i],
                    center_position=center_positions[i]) for i in range(len(df))
            ]


def read_dataset():
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


def load_local_dataset() -> dict:
    with open(CONNECTOME_BASE_PATH, 'rb') as f:
        return pickle.load(f)


read_dataset()

# conn = load_local_dataset()
# print(conn.keys())
