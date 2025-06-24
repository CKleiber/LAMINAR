import numpy as np
import matplotlib.pyplot as plt
import LAMINAR
import torch

# import tsne
from sklearn.manifold import TSNE
import umap.umap_ as umap


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
    if counts_sorted_cum[i] < 0.6:
        print(f"Label {counts_sorted[i]}: {counts[counts_sorted[i]]} points")
        l.append(counts_sorted[i])


# data_red is data with only the points in labels l
data_red = data[np.isin(labels, l)]

data_red = torch.tensor(data_red, dtype=torch.float32).to(device)   # ON GPU
#data_red = (data_red-data_red.mean())/data_red.std()

#standardize
data_red = (data_red - data_red.mean(dim=0)) / data_red.std(dim=0)  

# random permutation of data_red
perm = torch.randperm(data_red.size(0))
data_red = data_red[perm]
labels_red = labels[np.isin(labels, l)][perm]

#####
data_red = data_red[:1001]
labels_red = labels_red[:1001]
#####

# print number of points in each label
counts_red = np.bincount(labels_red)
# write in txt file
with open('counts_red.txt', 'w') as f:
    for i in range(len(l)):
        f.write(f"Label {l[i]}: {counts_red[l[i]]} points\n")

for i in range(len(l)):
    print(f"Label {l[i]}: {counts_red[l[i]]} points")


# plot the data for labels l in a 8 by 8 subplot grid
fig, axs = plt.subplots(8, 8, figsize=(20, 20))

for i in range(8):
    for j in range(8):
        for k in range(len(l)):
            axs[i, j].scatter(data_red[labels_red == l[k], i].cpu().numpy(), data_red[labels_red == l[k], j].cpu().numpy(), s=1, alpha=0.05)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

#plt.show()

# save the figure
print('Save Visualisation of Data')
plt.savefig('Data.png', dpi=300)


# tSNE does only work with CPU data
# apply t-SNE to data_red
print('Start tSNE Euclidean')

for perp in [5, 25, 50, 100, 250, 500, 1000]:
    tsne = TSNE(n_components=2, random_state=0, perplexity=perp, init='random')
    data_red_tsne = tsne.fit_transform(data_red.cpu().numpy())

    # plot the t-SNE data
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(l)):
        ax.scatter(data_red_tsne[labels_red[np.isin(labels_red, l)] == l[i], 0], data_red_tsne[labels_red[np.isin(labels_red, l)] == l[i], 1], s=1, alpha=0.25)
        ax.set_xticks([])
        ax.set_yticks([])
    #plt.show()

    # save the plot as a png file
    fig.savefig(f'tsne_eucl_{perp}.png', dpi=300)

# start UMAP
print('Start UMAP')
for neighbors in [5, 10, 25, 50, 100, 250, 500, 1000]:
    reducer = umap.UMAP(n_neighbors=neighbors, n_components=2, random_state=0)
    data_red_umap = reducer.fit_transform(data_red.cpu().numpy())
    # plot the UMAP data
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(len(l)):
        ax.scatter(data_red_umap[labels_red[np.isin(labels_red, l)] == l[i], 0], data_red_umap[labels_red[np.isin(labels_red, l)] == l[i], 1], s=1, alpha=0.25)
        ax.set_xticks([])
        ax.set_yticks([])
    #plt.show()

    # save the plot as a png file
    fig.savefig(f'umap_{neighbors}.png', dpi=300)

# train laminar on data_red
print('Start Training')
LAM = LAMINAR.LAMINAR(data_red, drop_freq=9999, epochs=2500, lr=0.01, save_distance_matrix=False) #, nTh=5, m=128)
pushed = LAM.X_pushed.cpu().numpy()
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
#plt.show()
fig.savefig('laminar_pushed.png', dpi=300)


print('Get Distance Matrix')
dists_matrix, _ = LAMINAR.utils.dijkstra.dijkstra(LAM.graph, [i for i in range(data_red.shape[0])], [i for i in range(data_red.shape[0])])

dists_matrix = np.array(dists_matrix, dtype=np.float32)


# tsne
for perp in [5, 25, 50, 100, 250, 500, 1000]:
    print(f'tSNE for LAMINAR with perpexity {perp}')

    tsne = TSNE(n_components=2, random_state=0, perplexity=perp, metric='precomputed', init='random')
    dists_matrix_tsne = tsne.fit_transform(dists_matrix)

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(len(l)):
        ax.scatter(dists_matrix_tsne[labels_red == l[i], 0], dists_matrix_tsne[labels_red == l[i], 1], s=1, alpha=0.25)
        ax.set_xticks([])
        ax.set_yticks([])

    #plt.show()

    # save the plot as a png file
    fig.savefig(f'tsne_lam_{perp}.png', dpi=300)

# umap

for neighbors in [5, 10, 25, 50, 100, 250, 500, 1000]:
    print(f'UMAP for LAMINAR with neighbors {neighbors}')

    reducer = umap.UMAP(n_neighbors=neighbors, n_components=2, random_state=0, metric='precomputed')
    dists_matrix_umap = reducer.fit_transform(dists_matrix)

    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(len(l)):
        ax.scatter(dists_matrix_umap[labels_red == l[i], 0], dists_matrix_umap[labels_red == l[i], 1], s=1, alpha=0.25)
        ax.set_xticks([])
        ax.set_yticks([])

    #plt.show()

    # save the plot as a png file
    fig.savefig(f'umap_lam_{neighbors}.png', dpi=300)
