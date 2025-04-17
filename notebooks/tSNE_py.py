import numpy as np
import matplotlib.pyplot as plt
import LAMINAR
import torch

# import tsne
from sklearn.manifold import TSNE


# CHECK IF USING GPU!
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")    

print('Load Data')
data = np.load('notebooks/newhalo_young_data_points.npy')
labels = np.load('notebooks/newhalo_young_satellite_labels.npy')
print('Data Loaded')

# check how many points are in each label and print in decreasing order
counts = np.bincount(labels)
counts_sorted = np.argsort(counts)[::-1]

#for i in counts_sorted:
#    print(f"Label {i}: {counts[i]} points")

counts_sorted_cum = np.cumsum(counts[counts_sorted])/len(data)

# print all labels until cumulative count is greater than 0.6
l = []
for i in range(max(labels)):
    if counts_sorted_cum[i] < 0.5:
        print(f"Label {counts_sorted[i]}: {counts[counts_sorted[i]]} points")
        l.append(counts_sorted[i])


# plot the data for labels l in a 8 by 8 subplot grid
fig, axs = plt.subplots(8, 8, figsize=(20, 20))

for i in range(8):
    for j in range(8):
        for k in range(len(l)):
            axs[i, j].scatter(data[labels == l[k], i], data[labels == l[k], j], s=1, alpha=0.01)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

plt.show()

# save the figure
print('Save Visualisation of Data')
plt.savefig('Data.png', dpi=300)

# data_red is data with only the points in labels l
data_red = data[np.isin(labels, l)]


'''
# tSNE does only work with CPU data
# apply t-SNE to data_red
print('Start tSNE Euclidean')
tsne = TSNE(n_components=2, random_state=0, perplexity=250)
data_red_tsne = tsne.fit_transform(data_red)

# plot the t-SNE data
fig, ax = plt.subplots(figsize=(10, 10))
for i in range(len(l)):
    ax.scatter(data_red_tsne[labels[np.isin(labels, l)] == l[i], 0], data_red_tsne[labels[np.isin(labels, l)] == l[i], 1], s=1, alpha=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# save the plot as a png file
fig.savefig('tsne_eucl_250.png', dpi=300)

'''


data_red = torch.tensor(data_red, dtype=torch.float32).to(device)   # ON GPU
#data_red = (data_red-data_red.mean())/data_red.std()

#standardize
data_red = (data_red - data_red.mean(dim=0)) / data_red.std(dim=0)  

# random permutation of data_red
perm = torch.randperm(data_red.size(0))
data_red = data_red[perm]
labels_red = labels[np.isin(labels, l)][perm]

#####
data_red = data_red[:1000]
labels_red = labels_red[:1000]
#####

# train laminar on data_red
print('Start Training')
LAM = LAMINAR.LAMINAR(data_red, drop_freq=9999, epochs=2500, lr=0.01, save_distance_matrix=False) #, nTh=5, m=128)
pushed = LAM.X_pushed
fig, axs = plt.subplots(8, 8, figsize=(30, 30))

print('Plot the result of LAMINAR learning')
for i in range(8):
    for j in range(8):
        axs[i, j].scatter(pushed[:, i], pushed[:, j], s=5, alpha=0.1)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].set_xlim(-1, 1)
        axs[i, j].set_ylim(-1, 1)

# save the plot as a png file
plt.show()
fig.savefig('laminar_pushed.png', dpi=300)


print('Get Distance Matrix')
dists_matrix, _ = LAMINAR.utils.dijkstra.dijkstra(LAM.graph, [i for i in range(data_red.shape[0])], [i for i in range(data_red.shape[0])])

dists_matrix = np.array(dists_matrix, dtype=np.float32)

for perp in [50, 100, 250, 500, 1000]:
    print(f'tSNE for LAMINAR with perpexity {perp}')

    tsne = TSNE(n_components=2, random_state=0, perplexity=perp, metric='precomputed', init='random')
    dists_matrix_tsne = tsne.fit_transform(dists_matrix)

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(len(l)):
        ax.scatter(dists_matrix_tsne[labels_red == l[i], 0], dists_matrix_tsne[labels_red == l[i], 1], s=1, alpha=1)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

    # save the plot as a png file
    fig.savefig(f'tsne_lam_{perp}.png', dpi=300)
