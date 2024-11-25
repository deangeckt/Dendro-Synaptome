import numpy as np

from synapse import Synapse
from connectome_types import ClfType


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
        return (f"Neuron(root_id={self.root_id}, cell_type='{self.cell_type}',"
                f"mtype={self.mtype}, position={self.position},"
                f"volume={self.volume},"
                f"#pre_synapses={len(self.pre_synapses)},"
                f"#post_synapses={len(self.post_synapses)})")
