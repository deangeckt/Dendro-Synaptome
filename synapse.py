import numpy as np


class Synapse:
    def __init__(self,
                 id_: int,
                 pre_pt_root_id: int,
                 post_pt_root_id: int,
                 size: int,
                 center_position: np.ndarray):

        self.id_ = id_
        self.pre_pt_root_id = pre_pt_root_id
        self.post_pt_root_id = post_pt_root_id
        self.size = size
        self.center_position = center_position

    def __repr__(self):
        return (f"Synapse(id={self.id_}, pre_pt_root_id={self.pre_pt_root_id}, "
                f"post_pt_root_id={self.post_pt_root_id}, size={self.size}, "
                f"center_position={self.center_position})")
