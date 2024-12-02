from enum import Enum

from synapse import Synapse

DATA_BASE_PATH = 'data/'
CONNECTOME_BASE_PATH = 'data/connectome_base.pkl'
CONNECTOME_TOY_PATH = 'data/connectome_toy.pkl'
SKELETONS_DIR_PATH = 'data/skeletons'
NEURONS_PATH = 'data/neurons'


class SynapseDirection(str, Enum):
    input = 'input'
    output = 'output'


class ClfType(str, Enum):
    excitatory = 'E'
    inhibitory = 'I'


# morphological types
m_types = ['DTC',
           'ITC',
           'L2a',
           'L2b',
           'L2c',
           'L3a',
           'L3b',
           'L4a',
           'L4b',
           'L4c',
           'L5ET',
           'L5NP',
           'L5a',
           'L5b',
           'L6short-a',
           'L6short-b',
           'L6tall-a',
           'L6tall-b',
           'L6tall-c',
           'PTC',
           'STC']

cell_types = ['23P',
              '4P',
              '5P-ET',
              '5P-IT',
              '5P-NP',
              '6P-CT',
              '6P-IT',
              'BC',
              'BPC',
              'MC',
              'NGC',
              'OPC',
              'astrocyte',
              'microglia',
              'oligo',
              'pericyte']
