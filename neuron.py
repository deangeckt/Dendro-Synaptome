import json
import os

import numpy as np
from typing import Optional

from connectome_types import SKELETONS_DIR_PATH
from synapse import Synapse
from connectome_types import ClfType
from meshparty.skeleton import Skeleton
import meshparty.skeleton_io


class Neuron:
    def __init__(self,
                 root_id: int,
                 clf_type: ClfType,
                 cell_type: str,
                 mtype: int,
                 position: np.ndarray,
                 volume: float,
                 pre_synapses: list[Synapse],
                 post_synapses: list[Synapse]):

        self.root_id = root_id
        self.clf_type = clf_type
        self.cell_type = cell_type
        self.mtype = mtype
        self.position = position
        self.volume = volume
        self.pre_synapses = pre_synapses
        self.post_synapses = post_synapses

    def __repr__(self):
        return (f"Neuron(root_id={self.root_id},"
                f"clf_type={self.clf_type.value},"
                f"cell_type={self.cell_type},"
                f"mtype={self.mtype},"
                f"position={self.position},"
                f"volume={self.volume},"
                f"#pre_synapses={len(self.pre_synapses)},"
                f"#post_synapses={len(self.post_synapses)})")

    def load_skeleton(self) -> Optional[Skeleton]:
        cell_id = self.root_id
        sk_file_path = f'{SKELETONS_DIR_PATH}/{cell_id}.json'
        if not os.path.exists(sk_file_path) or os.path.getsize(sk_file_path) == 0:
            return None

        with open(sk_file_path) as f:
            sk_dict = json.load(f)

            return meshparty.skeleton.Skeleton(
                vertices=np.array(sk_dict['vertices']),
                edges=np.array(sk_dict['edges']),
                mesh_to_skel_map=sk_dict['mesh_to_skel_map'],
                vertex_properties=sk_dict['vertex_properties'],
                root=sk_dict['root'],
                meta=sk_dict['meta'],
            )