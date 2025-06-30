import numpy as np
import matplotlib.pyplot as plt


'''
Convert the metric tensor to a colour in the HSV space. 
This allowes the visualisation of the metric tensors in a 2D space, 
where the hue corresponds to the angle of the eigenvector with the largest eigenvalue (= orientation of the ellipse),
and the saturation corresponds to the ratio between the eigenvalues (= shape of the ellipse).
'''


# convert HSV to RGB which is used by matplotlib
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


# plot a reference circle with all the colours
# this is used to see which colour corresponds to which angle and saturation
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


# convert the tensors to a colour in the HSV space
def get_cov_colours(
            data,               # original data in original space
            metric_tensor             
            ):
    colours = []

    for point in range(len(data)):
        # get the eigenvalues and eigenvectors of the metric tensor at the point
        eigenvalues, eigenvectors = np.linalg.eig(metric_tensor[point])

        # volume of the ellipse
        #volume = np.sqrt(np.prod(eigenvalues))

        # scale it so a large value goes asymptotically to 1
        # can also be used for visualisation:
        #V = 1 - np.exp(-volume)
        
        # get angle
        eig_idx = np.argmax(eigenvalues)
        vec = eigenvectors[:, eig_idx]

        angle = np.arctan2(vec[1], vec[0])
        angle = np.degrees(angle)

        # make sure the angle is in the range [0, 360)
        if angle < 0:
            angle += 360

        colour_angle = (angle % 180) * 2
        H = colour_angle

        # get the ratio
        ratio = eigenvalues.max() / (eigenvalues.min())
        
        # ratio of 1 means no saturation i.e. S = 0 (circle)
        # ratio of inf means full saturation i.e. S = 1 (very stretched ellipse)
        S = 1 - (1/ratio)**0.25
        
        R, G, B = hsv_to_rgb(H, S, 1)
        colours.append((R, G, B))
    return colours