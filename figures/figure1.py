import pylustrator
import pickle

from matplotlib.colors import LogNorm

# from connectivity matrix notebook
with open('figure1.pkl', 'rb') as file:
    fig1_data = pickle.load(file)

print(fig1_data.keys())


def get_none_empty_mat(z):
    non_empty_rows = ~np.all(z == 0, axis=1)
    non_empty_cols = ~np.all(z == 0, axis=0)
    return z[non_empty_rows][:, non_empty_cols]


def plot_cell_type_conn_matrix(z, labels, title, ax, remove=[], log_scale=True, text_inside=False):
    if ax is None:
        _, ax = plt.subplots()

    non_empty_rows = ~np.all(z == 0, axis=1)
    z_filtered = get_none_empty_mat(z)
    labels_filtered = np.array(labels)[non_empty_rows]

    if remove:
        mask = ~np.isin(labels_filtered, remove)
        z_filtered = z_filtered[mask][:, mask]
        labels_filtered = labels_filtered[mask]

    if log_scale:
        z_filtered = z_filtered + 1e-10
        c = ax.imshow(z_filtered, cmap='YlOrRd', norm=LogNorm(vmin=z_filtered.min(), vmax=z_filtered.max()))
        cbar = plt.colorbar(c, ax=ax)
        cbar.set_label('Log scale')
    else:
        c = ax.imshow(z_filtered, cmap='YlOrRd')
        cbar = plt.colorbar(c, ax=ax)

    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels_filtered)))
    ax.set_xticklabels(labels_filtered, rotation=90)
    ax.set_yticks(np.arange(len(labels_filtered)))
    ax.set_yticklabels(labels_filtered)
    ax.set_ylabel('Post-synaptic neuron type')
    ax.set_xlabel('Pre-synaptic neuron type')

    if text_inside:
        for i in range(z_filtered.shape[0]):
            for j in range(z_filtered.shape[1]):
                ax.text(j, i, f'{z_filtered[i, j]:.2f}', ha='center', va='center', color='black')

    return ax


import os

# running from Root directory
os.chdir("..")

from connectome_types import ClfType, CONNECTOME_NEURON_TABLE_PATH
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# pylustrator.start()

clf_type_space = [e.value for e in ClfType]

neurons_df = pd.read_csv(CONNECTOME_NEURON_TABLE_PATH)
neurons_df_sorted = neurons_df.sort_values('cell_type')
ex_color = "#ff0000"
inh_color = "#0072BD"

ei_palette = {"I": inh_color, "E": ex_color}
ei_combine_palette = {"I": inh_color, "E": ex_color, "All": "lightgray"}

# cell types colors
cell_types = sorted(neurons_df['cell_type'].unique())
combine_cell_type_order = cell_types.copy()
combine_cell_type_order += ['E', 'I', 'All']

data_cols = ['ds_num_of_incoming_synapses', 'ds_num_of_outgoing_synapses',
             'ds_incoming_syn_mean_weight', 'ds_outgoing_syn_mean_weight',
             'ds_incoming_syn_sum_weight', 'ds_outgoing_syn_sum_weight'
             ]

all_type_df = neurons_df[data_cols].copy()
all_type_df['type_'] = ['All'] * len(all_type_df)
all_type_df['clf_type'] = ['All'] * len(all_type_df)

cell_type_df = neurons_df[['cell_type', 'clf_type'] + data_cols].copy()
cell_type_df = cell_type_df.rename(columns={'cell_type': 'type_'})

clf_type_df = neurons_df[['clf_type'] + data_cols].copy()
clf_type_df['type_'] = clf_type_df['clf_type'].copy()
clf_type_df.head()

combined_df = pd.concat([cell_type_df, clf_type_df, all_type_df], ignore_index=True)
combined_df['type_'] = pd.Categorical(combined_df['type_'], categories=combine_cell_type_order, ordered=True)

comb_cell_type_counts = combined_df['type_'].value_counts()
comb_cell_type_counts_labels = [f"{ct} (N={comb_cell_type_counts[ct]})" for ct in combine_cell_type_order]

fig = plt.figure(figsize=(15, 15))

gs = fig.add_gridspec(4, 3, hspace=0.4)
ax0 = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[1, 2])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])
ax6 = fig.add_subplot(gs[2, 2])
ax7 = fig.add_subplot(gs[3, 0])
ax8 = fig.add_subplot(gs[3, 1])
ax9 = fig.add_subplot(gs[3, 2])
plt.tight_layout()
from connectome_types import cell_types as cell_types_orig

celltype_mat = fig1_data['conn_mat']
celltype_mat_size = fig1_data['weight_mat']
celltype_mat_size_avg = fig1_data['avg_weight_mat']

# Plot 0
h = sns.histplot(data=neurons_df_sorted, x="cell_type", hue="clf_type",
                 palette=ei_palette, alpha=1, ax=ax0, multiple="stack")
for container in h.containers:
    h.bar_label(container, labels=[''] * len(container))
for container in h.containers:
    ax0.bar_label(container, fontsize=10, padding=2, label_type='edge')

ax0.set_ylim(top=ax0.get_ylim()[1] * 1.1)
ax0.set_title("Neuron type distribution (total: 63904 E, 7832 I)")
# ax0.set_xticklabels(ax0.get_xticklabels(), ha='right') #rotation=45,
# ax0.set_xlabel("Cell type")
sns.move_legend(ax0, title='Type', loc='best')

## ROW 1
sns.boxplot(y="type_", x="ds_num_of_incoming_synapses", hue="clf_type", data=combined_df,
            palette=ei_combine_palette, showfliers=False, ax=ax1)
ax1.set_title("In degree distribution")
ax1.set_xlabel("Number of connections")
ax1.legend(title="Type")

sns.boxplot(y="type_", x="ds_num_of_outgoing_synapses", hue="clf_type", data=combined_df,
            palette=ei_combine_palette, showfliers=False, ax=ax2)
ax2.set_title("Out degree distribution")
ax2.set_xlabel("Number of connections")

plot_cell_type_conn_matrix(celltype_mat, cell_types_orig, 'Total number of connections', ax=ax3, log_scale=True)

## ROW 2
sns.boxplot(y="type_", x="ds_incoming_syn_sum_weight", hue="clf_type", data=combined_df,
            palette=ei_combine_palette, showfliers=False, ax=ax4)
ax4.set_title("Incoming sum synaptic cleft size")
ax4.set_xlabel("Sum of synaptic cleft size (voxels)")

sns.boxplot(y="type_", x="ds_outgoing_syn_sum_weight", hue="clf_type", data=combined_df,
            palette=ei_combine_palette, showfliers=False, ax=ax5)
ax5.set_title("Outgoing sum synaptic cleft size")
ax5.set_xlabel("Sum of synaptic cleft size (voxels)")

plot_cell_type_conn_matrix(celltype_mat_size, cell_types_orig, 'Total sum of weights', ax=ax6, log_scale=True)

## ROW 3

sns.boxplot(y="type_", x="ds_incoming_syn_mean_weight", hue="clf_type", data=combined_df,
            palette=ei_combine_palette, showfliers=False, ax=ax7)
ax7.set_title("Incoming mean synaptic cleft size")
ax7.set_xlabel("Mean synaptic cleft size (voxels)")

sns.boxplot(y="type_", x="ds_outgoing_syn_mean_weight", hue="clf_type", data=combined_df,
            palette=ei_combine_palette, showfliers=False, ax=ax8)
ax8.set_title("Outgoing mean synaptic cleft size")
ax8.set_xlabel("Mean synaptic cleft size (voxels)")

# this is filtered already - use cell_type which is filtered
plot_cell_type_conn_matrix(celltype_mat_size_avg, cell_types, 'Mean weights', ax=ax9, log_scale=False)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(34.990000/2.54, 33.490000/2.54, forward=True)
# plt.figure(1).ax_dict["<colorbar>"].set_position([0.893071, 0.538939, 0.007941, 0.162330])
# plt.figure(1).ax_dict["<colorbar>"].set_position([0.893071, 0.311676, 0.007941, 0.162330])
# plt.figure(1).ax_dict["<colorbar>"].set_position([0.893071, 0.084414, 0.007941, 0.162330])
plt.figure(1).axes[0].set(position=[0.2161, 0.7319, 0.6577, 0.1481])
plt.figure(1).axes[0].set_position([0.196182, 0.766201, 0.705467, 0.162330])
plt.figure(1).axes[0].get_legend().set(visible=False)
plt.figure(1).axes[1].legend(loc=(-0.05543, 1.397), title='Type')
plt.figure(1).axes[1].set_position([0.098518, 0.538939, 0.244478, 0.162330])
plt.figure(1).axes[1].patches[0].set_facecolor("#ff0000")
plt.figure(1).axes[1].patches[1].set_facecolor("#ff0000")
plt.figure(1).axes[1].patches[2].set_facecolor("#ff0000")
plt.figure(1).axes[1].patches[3].set_facecolor("#ff0000")
plt.figure(1).axes[1].patches[4].set_facecolor("#ff0000")
plt.figure(1).axes[1].patches[5].set_facecolor("#ff0000")
plt.figure(1).axes[1].patches[6].set_facecolor("#ff0000")
plt.figure(1).axes[1].patches[7].set_facecolor("#ff0000")
plt.figure(1).axes[1].patches[14].set_facecolor("#ff0000")
plt.figure(1).axes[1].get_yaxis().get_label().set(text='')
plt.figure(1).axes[2].set_position([0.391891, 0.538939, 0.244478, 0.162330])
plt.figure(1).axes[2].get_legend().set(visible=False)
plt.figure(1).axes[2].patches[0].set_facecolor("#ff0000")
plt.figure(1).axes[2].patches[1].set_facecolor("#ff0000")
plt.figure(1).axes[2].patches[2].set_facecolor("#ff0000")
plt.figure(1).axes[2].patches[3].set_facecolor("#ff0000")
plt.figure(1).axes[2].patches[4].set_facecolor("#ff0000")
plt.figure(1).axes[2].patches[5].set_facecolor("#ff0000")
plt.figure(1).axes[2].patches[6].set_facecolor("#ff0000")
plt.figure(1).axes[2].patches[7].set_facecolor("#ff0000")
plt.figure(1).axes[2].patches[14].set_facecolor("#ff0000")
plt.figure(1).axes[2].get_yaxis().get_label().set(text='')
plt.figure(1).axes[3].set_position([0.725577, 0.538957, 0.155270, 0.162294])
plt.figure(1).axes[3].xaxis.labelpad = -3.411765
plt.figure(1).axes[3].get_xaxis().get_label().set(text='')
plt.figure(1).axes[4].set_position([0.098518, 0.311676, 0.244478, 0.162330])
plt.figure(1).axes[4].get_legend().set(visible=False)
plt.figure(1).axes[4].patches[0].set_facecolor("#ff0000")
plt.figure(1).axes[4].patches[1].set_facecolor("#ff0000")
plt.figure(1).axes[4].patches[2].set_facecolor("#ff0000")
plt.figure(1).axes[4].patches[3].set_facecolor("#ff0000")
plt.figure(1).axes[4].patches[4].set_facecolor("#ff0000")
plt.figure(1).axes[4].patches[5].set_facecolor("#ff0000")
plt.figure(1).axes[4].patches[6].set_facecolor("#ff0000")
plt.figure(1).axes[4].patches[7].set_facecolor("#ff0000")
plt.figure(1).axes[4].patches[14].set_facecolor("#ff0000")
plt.figure(1).axes[4].get_yaxis().get_label().set(text='')
plt.figure(1).axes[5].set_position([0.391891, 0.311676, 0.244478, 0.162330])
plt.figure(1).axes[5].yaxis.labelpad = -4.000000
plt.figure(1).axes[5].get_legend().set(visible=False)
plt.figure(1).axes[5].patches[0].set_facecolor("#ff0000")
plt.figure(1).axes[5].patches[1].set_facecolor("#ff0000")
plt.figure(1).axes[5].patches[2].set_facecolor("#ff0000")
plt.figure(1).axes[5].patches[3].set_facecolor("#ff0000")
plt.figure(1).axes[5].patches[4].set_facecolor("#ff0000")
plt.figure(1).axes[5].patches[5].set_facecolor("#ff0000")
plt.figure(1).axes[5].patches[6].set_facecolor("#ff0000")
plt.figure(1).axes[5].patches[7].set_facecolor("#ff0000")
plt.figure(1).axes[5].patches[14].set_facecolor("#ff0000")
plt.figure(1).axes[5].get_yaxis().get_label().set(text='')
plt.figure(1).axes[6].set_position([0.725577, 0.311694, 0.155270, 0.162294])
plt.figure(1).axes[6].get_xaxis().get_label().set(text='')
plt.figure(1).axes[7].set(position=[0.09852, 0.08441, 0.2445, 0.1623], xlim=(0., 10760.), ylabel='')
plt.figure(1).axes[7].get_legend().set(visible=False)
plt.figure(1).axes[7].patches[0].set_facecolor("#ff0000")
plt.figure(1).axes[7].patches[1].set_facecolor("#ff0000")
plt.figure(1).axes[7].patches[2].set_facecolor("#ff0000")
plt.figure(1).axes[7].patches[3].set_facecolor("#ff0000")
plt.figure(1).axes[7].patches[4].set_facecolor("#ff0000")
plt.figure(1).axes[7].patches[5].set_facecolor("#ff0000")
plt.figure(1).axes[7].patches[6].set_facecolor("#ff0000")
plt.figure(1).axes[7].patches[7].set_facecolor("#ff0000")
plt.figure(1).axes[7].patches[14].set_facecolor("#ff0000")
plt.figure(1).axes[7].get_yaxis().get_label().set(text='')
plt.figure(1).axes[8].set_position([0.391891, 0.084414, 0.244478, 0.162330])
plt.figure(1).axes[8].get_legend().set(visible=False)
plt.figure(1).axes[8].patches[0].set_facecolor("#ff0000")
plt.figure(1).axes[8].patches[1].set_facecolor("#ff0000")
plt.figure(1).axes[8].patches[2].set_facecolor("#ff0000")
plt.figure(1).axes[8].patches[3].set_facecolor("#ff0000")
plt.figure(1).axes[8].patches[4].set_facecolor("#ff0000")
plt.figure(1).axes[8].patches[5].set_facecolor("#ff0000")
plt.figure(1).axes[8].patches[6].set_facecolor("#ff0000")
plt.figure(1).axes[8].patches[7].set_facecolor("#ff0000")
plt.figure(1).axes[8].patches[14].set_facecolor("#ff0000")
plt.figure(1).axes[8].get_yaxis().get_label().set(text='')
plt.figure(1).axes[9].set_position([0.725577, 0.084432, 0.155270, 0.162294])
#% end: automatic generated code from pylustrator
plt.show()
