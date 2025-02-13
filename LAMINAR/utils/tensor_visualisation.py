import numpy as np
import matplotlib.pyplot as plt

def hsv_to_rgb(H, S, V):
    h = np.floor(H/60)
    f = H/60 - h
    p = V * (1 - S)
    q = V * (1 - f * S)
    t = V * (1 - (1 - f) * S)

    if (h == 0) or (h == 6):
        return V, t, p
    if h == 1:
        return q, V, p
    if h == 2:
        return p, V, t
    if h == 3:
        return p, q, V
    if h == 4:
        return t, p, V
    if h == 5:
        return V, p, q

# plot each colour on a circle where the angle corresponds to the hue
def plot_reference():
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for r in np.linspace(0, 1, 20):
        S = r
        for angle in range(360):
            H = angle * 2 % 360
            R, G, B = hsv_to_rgb(H, S, 1)
            ax.plot(r*np.cos(angle/180*np.pi), r*np.sin(angle/180*np.pi), marker='o', color=(R, G, B), markersize=10)
    # axis off
    ax.axis('off')
    plt.show()

# only two dim
def get_cov_colours(
            data,               # original data in original space
            metric_tensor             
            ):
    colours = []

    for point in range(len(data)):
        eigenvalues, eigenvectors = np.linalg.eig(metric_tensor[point])

        # volume of the ellipse
        volume = np.sqrt(np.prod(eigenvalues))

        # scale it so a large value goes asymptotically to 1
        S = 1 - np.exp(-volume)
        
        eig_idx = np.argmax(eigenvalues)
        vec = eigenvectors[:, eig_idx]

        angle = np.arctan2(vec[1], vec[0])
        angle = np.degrees(angle)

        if angle < 0:
            angle += 360

        colour_angle = (angle % 180) * 2

        H = colour_angle

        #print(eigenvalues)

        ratio = eigenvalues.max() / (eigenvalues.min())
        #print(ratio)

        # ratio of 1 means black i.e. V = 0
        # ratio of inf means white i.e. V = 1
        V = 1 - (1/ratio)**0.25
        #print(H, S)
        R, G, B = hsv_to_rgb(H, V, 1)
        colours.append((R, G, B))
    return colours