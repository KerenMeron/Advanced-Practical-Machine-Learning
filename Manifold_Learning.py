'''
Manifold Learning simulation to demonstrate different dimensionality reduction algorithms.
Advanced Practical Machine Learning, HUJI.
Keren Meron, Nov. 2017
'''

import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage.interpolation import rotate
from scipy.misc import imread


def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.ion()
    plt.show()


def swiss_roll_example():
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.ion()
    plt.show()


def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels**0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.ion()
    plt.show()


def plot_with_images(X, images, title, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param image_num: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)
    return fig


def scree_plot_example(sigma):
    '''
    Create a random 2-dimensional data set and embed it in a higher dimension using a random rotation
    matrix, then add Gaussian noise to your new data set. Test varying degrees of noise and see how
    the eigenvalues of MDS change as the noise increases.
    '''
    n, p = 10, 5
    d = 2  # any value

    # random data set of size nxp, of gaussian data padded with zeros
    data_set = np.hstack((np.random.rand(n, d), np.zeros((n, p-d))))

    # random rotation matrix: by taking Q from QR decomposition of a random gaussian matrix
    gauss_mat = np.random.normal(size=(p, p))
    rotation_matrix, _ = np.linalg.qr(gauss_mat)

    # embed data in higher dimension, and add gaussian noise
    higher_dim_data = data_set.dot(rotation_matrix.T)
    noise_matrix = np.random.normal(size=higher_dim_data.shape) * sigma
    higher_dim_data += noise_matrix

    # apply MDS on the high dimensional data, and show screeplot of eigenvalues
    _, eigvalues = MDS(higher_dim_data, d, True)

    plt.figure()
    title = 'Scree Plot\n'+(r'$\sigma=%.1f$' % sigma)
    plt.title(title)
    plt.xticks(np.arange(10)+1)
    plt.scatter(np.arange(10)+1, eigvalues[:10])
    plt.ion()
    plt.show()


def MDS(X, d, get_eigens=False):
    '''
    Given a nxp data matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: data matrix.
    :param d: the dimension.
    :param get_eigens: [bool] if true, return resulting eigenvalues as well
    :return: Nxd reduced data point matrix, eigenvalues of resulting data [optional]
    '''
    n = X.shape[0]
    euc_dist = euclidean_distances(X, X) ** 2

    H = np.identity(n) - np.ones((n,n)) / n
    S = -0.5 * np.dot(H, np.dot(euc_dist, H))

    # diagonalize
    S_eigen_values, S_eigen_vectors = np.linalg.eig(S)
    S_eigen_values = np.real(S_eigen_values)

    # if any negative eigen values, take abs value and multiply matching vector by -1
    eigen_values_matrix = np.diag(S_eigen_values)
    S_eigen_vectors[:, np.where(eigen_values_matrix < 0)[1]] *= -1
    S_eigen_values = np.abs(S_eigen_values)

    # get d largest eigen values and matching vectors
    largest_indices = np.argsort(-S_eigen_values)[:d]
    largest_eigen_values = S_eigen_values[largest_indices]
    if get_eigens:
        return None, S_eigen_values

    largest_eigen_vectors = S_eigen_vectors[:, largest_indices]
    result = largest_eigen_vectors * np.sqrt(largest_eigen_values)
    return result


def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''

    n, p = X.shape[0], X.shape[1]
    unit_vector = np.ones((1, k))
    if k >= n:
        k = n-1

    # find K nearest neighbors (in practice k+1 because closest neighbor is self)
    dists = euclidean_distances(X, X) ** 2
    KNN_indices = dists.argsort()[:, 1:k+1]

    # find matrix W which minimizes the residual sum of squares
    weights = np.zeros((n, n))
    for i in range(X.shape[0]):
        Xi_neighbors = X[KNN_indices[i]]
        L1_dist_neighbors = X[i] - Xi_neighbors
        gramm = np.dot(L1_dist_neighbors, L1_dist_neighbors.T)
        gramm_inv = np.linalg.pinv(gramm)
        Lambda = 2 / (unit_vector.dot(gramm_inv).dot(unit_vector.T))[0,0]
        weights[i, KNN_indices[i]] = (Lambda / 2) * np.dot(gramm_inv, unit_vector.T).flatten()

    # decompose into eigenvectors and return those corresponding to 2,...,d+1 lowest eigenvalues
    construct_M = np.identity(n) - weights
    M = np.dot(construct_M.T, construct_M)
    eig_values, eig_vectors = np.linalg.eig(M)
    eig_values = np.real(eig_values)

    # if any negative eigen values, take abs value and multiply matching vector by -1
    eigen_values_matrix = np.diag(eig_values)
    eig_vectors[:, np.where(eigen_values_matrix < 0)[1]] *= -1
    eig_values = np.abs(eig_values)

    smallest_eig_indices = np.argsort(eig_values)[1:d+1]
    return eig_vectors[:, smallest_eig_indices]


def DiffusionMap(X, d, sigma, t, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    gram matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the gram matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :param k: the amount of neighbors to take into account when calculating the gram matrix.
    :return: Nxd reduced data matrix.
    '''

    # create kernel similarity matrix, using heat kernel
    dists = euclidean_distances(X, X) ** 2
    n = X.shape[0]
    kernel_matrix = np.zeros((n, n))
    nearest_neigh_indices = dists.argsort()[:, 1:k+1]

    for i in range(n):
        kernel_matrix[i, nearest_neigh_indices[i]] = np.exp(-dists[i, nearest_neigh_indices[i]]/sigma)

    # normalize the kernel
    row_sums = np.sum(kernel_matrix, axis=1)
    row_sums_diag = np.diag(row_sums)
    markov_transition_matrix = np.linalg.pinv(row_sums_diag).dot(kernel_matrix)

    # extract eigenvectors corresponding to 2,...,d+1 highest eigenvalues
    eig_values, eig_vectors = np.linalg.eig(markov_transition_matrix)

    # if any negative eigen values, take abs value and multiply matching vector by -1
    eigen_values_matrix = np.diag(eig_values)
    eig_vectors[:, np.where(eigen_values_matrix < 0)[1]] *= -1
    eig_values = np.abs(eig_values)

    highest_eig_indices = np.argsort(eig_values)[-d-1:-1][::-1]
    highest_eig_vectors = eig_vectors[:, highest_eig_indices]
    return highest_eig_vectors * (eig_values[highest_eig_indices] ** t)


def display_swiss_roll(func, **kwargs):
    '''
    Perform different manifold dim reduction algorithms on swiss roll data. Plot results.
    :param func: function to perform on data, implementing algorithm. One of: MDS, LLE, DiffusionMap
    :param kwargs: args to pass to func 
    '''
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)
    kwargs.update({'X': X})
    reduced_data = func(**kwargs)
    plt.figure()
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title('{}  k={} sigma={} t={}'.format(func.__name__, kwargs.get('k'), kwargs.get('sigma'), kwargs.get('t')))
    plt.ion()
    plt.show()


def display_mnist_digits(func, **kwargs):
    '''
    Perform different manifold dim reduction algorithms on MNIST digits data. Plot results.
    :param func: function to perform on data, implementing algorithm. One of: MDS, LLE, DiffusionMap
    :param kwargs: args to pass to func 
    '''
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target
    colors = ['blue', 'plum', 'green', 'aqua', 'gray', 'orange', 'red', 'purple', 'pink', 'brown']

    kwargs.update({'X': data})
    reduced_data = func(**kwargs)

    plt.figure()
    for i in range(reduced_data.shape[0]):
        plt.text(reduced_data[i, 0], reduced_data[i, 1], str(labels[i]), color=colors[labels[i]])
        # invisible plot in order to center figure around data
        plt.plot(reduced_data[i, 0], reduced_data[i, 1], alpha=0)
        plt.title(func.__name__)
        plt.axis('off')
    plt.ion()
    plt.show()


def display_face_recognition(path, func, **kwargs):
    '''
    Perform different manifold dim reduction algorithms on faces data. Plot results.
    :param func: function to perform on data, implementing algorithm. One of: MDS, LLE, DiffusionMap
    :param kwargs: args to pass to func 
    '''
    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, _ = np.shape(X)

    kwargs.update({'X': X})
    reduced_data = func(**kwargs)

    plot_with_images(reduced_data, X, 'Faces Recognition', num_images)
    plt.title('{}  k={} t={} sigma={}'.format(func.__name__, kwargs.get('k'),kwargs.get('t'),kwargs.get('sigma')))
    plt.ion()
    plt.show()


def distances_compression_comparison():
    '''
    See how dimensionality reduction affects distances in-between data.
    Plot pairwise distances before and after dimensionality reduction.
    '''
    n = 30
    dimensions = [(50, 50), (50, 25), (50, 15), (50, 5)]
    plt.figure()
    for i, (p, d) in enumerate(dimensions):
        data = np.random.normal(size=(n, p))
        distances = euclidean_distances(data, data)
        distances = np.triu(distances).flatten()
        reduced_data = MDS(data, d)
        reduced_distances = euclidean_distances(reduced_data, reduced_data)
        reduced_distances = np.triu(reduced_distances).flatten()

        plt.subplot(2, 2, i+1)
        plt.scatter(distances, reduced_distances)
        plt.title('Distances Compression\nd={} p={}'.format(d, p))
        plt.xlabel('original data distances')
        plt.ylabel('reduced data distances')
    plt.ion()
    plt.show()


def rotate_and_get_intrinsic_struct(pic_path, func, **kwargs):
    '''
    Rotate given picture in many different angles.
    Then use dim reduction algorithms to find intrinsic structure.
    :param pic_path: path to file containing image in RGB form.
    :param func: function to use on data (for algorithm)
    :param kwargs: args needed for func
    '''
    # rotate image
    img_data = imread(pic_path, mode='L')

    rotated_images = np.array([copy.deepcopy(img_data)]).flatten().reshape(1, img_data.size)
    for angle in range(0, 360, 1):
        rotated = np.array([rotate(img_data, angle, reshape=False)]).flatten().reshape(1, img_data.size)
        rotated_images = np.concatenate((rotated_images, rotated), axis=0)
    kwargs.update({'X': rotated_images})
    reduced_data = func(**kwargs)

    plot_with_images(reduced_data, rotated_images, 'Pac Man', rotated_images.shape[0])
    plt.title('{}  k={}'.format(func.__name__, kwargs.get('k')))
    plt.ion()
    plt.show()


if __name__ == '__main__':

    # parameters for the different algorithms
    # perform here fine tuning of parameters
    mds_params = dict(d=2)
    lle_params = dict(d=2, k=10)
    diffmap_params = dict(d=2, sigma=60, t=20, k=10)

    '''
    Simulation Instructions:
    Please comment out any of the following lines which you do not wish to run.
    In order to change the data parameters, please edit above lines.
    '''

    # run screeplot to choose best d parameter
    for sigma in [0.1, 0.3, 0.5, 0.7, 0.9]:
        scree_plot_example(sigma)

    # run algorithms on swiss roll data
    display_swiss_roll(MDS, **mds_params)
    display_swiss_roll(LLE, **lle_params)
    display_swiss_roll(DiffusionMap, **diffmap_params)

    # run algorithms on mnist digits
    display_mnist_digits(MDS, **mds_params)
    display_mnist_digits(LLE, **lle_params)
    display_mnist_digits(DiffusionMap, **diffmap_params)

    # run algorithms on faces data
    display_face_recognition('faces.pickle', MDS, **mds_params)
    display_face_recognition('faces.pickle', LLE, **lle_params)
    display_face_recognition('faces.pickle', DiffusionMap, **diffmap_params)

    # show lossy compression of distances
    distances_compression_comparison()

    # bonus pac man
    rotate_and_get_intrinsic_struct('Pac Man.jpg', MDS, **mds_params)
    rotate_and_get_intrinsic_struct('Pac Man.jpg', LLE, **lle_params)
    rotate_and_get_intrinsic_struct('Pac Man.jpg', DiffusionMap, **diffmap_params)

    plt.show(block=True)
