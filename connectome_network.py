from collections import Counter

import networkx as nx
from matplotlib import pyplot as plt

from connectome import Connectome


class ConnectomeNetwork:
    def __init__(self, connectome: Connectome):
        self.connectome = connectome

        synapses_edges = [(s.pre_pt_root_id, s.post_pt_root_id) for s in connectome.synapses]
        edges = [(src, tar, count) for (src, tar), count in Counter(synapses_edges).items()]

        self.graph = nx.DiGraph()
        for src, tar, count in edges:
            self.graph.add_edge(src, tar, count=count)

        print('Graph:')
        print(f'\t#edges: {len(edges)}')
        print(f'\t#nodes: {len(self.graph.nodes)}')

    def basic_graph_plot(self):
        pos = nx.circular_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=False, node_color='lightblue', node_size=500, arrowsize=20)
        edge_labels = nx.get_edge_attributes(self.graph, 'count')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()
