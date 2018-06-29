'''
APML: Clutering.
Keren Meron 200039626
'''

import copy
import pickle
import numpy as np
from sklearn import datasets
from scipy.misc import imread
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise
from scipy.ndimage.interpolation import rotate


def circles_example():
    """
    an example function for generating and plotting synthesised data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)
    return circles

    # plt.plot(circles[0,:], circles[1,:], '.k')
    # plt.show()


def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    # plt.plot(apml[:, 0], apml[:, 1], '.')
    # plt.show()
    return apml


def microarray_exploration(data_path='microarray_data.pickle',
                            genes_path='microarray_genes.pickle',
                            conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    # plt.figure()
    # plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    # plt.show()

    # look at two conditions and their correlation:
    # plt.figure()
    # plt.scatter(data[:, 27], data[:, 29])
    # plt.plot([-5,5],[-5,5],'r')
    # plt.show()

    # see correlations between conds:
    # correlation = np.corrcoef(np.transpose(data))
    # plt.figure()
    # plt.imshow(correlation, extent=[0, 1, 0, 1])
    # plt.colorbar()
    # plt.show()

    # look at the entire data set:
    # plt.figure()
    # plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    # plt.colorbar()
    # plt.show()
    return data


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    return pairwise.euclidean_distances(X, Y)


def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster [shape X.shape[1])
    """
    return np.average(X, axis=0)


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix, Nxd
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    centers = np.zeros((k, X.shape[1]))
    indices_centers = np.zeros(k).astype(int)

    # choose first center uniformly random from data
    idx_chosen = np.random.choice(np.arange(X.shape[0]))
    indices_centers[0] = idx_chosen
    centers[0] = X[idx_chosen]
    centers_chosen = 1

    while centers_chosen < k:

        # choose center from remaining data points
        remaining_data = np.delete(X, indices_centers, axis=0)
        # compute distance of each point in data set from centers
        distances = metric(remaining_data, centers[:centers_chosen])  # Nxk

        # extract minimal distance for each point (for closest center)
        min_distances = np.min(distances, axis=1)[:, None]  # Nx1
        D_squared = np.square(min_distances)  # Nx1

        # choose center randomly with weighted probability ~ D_squared
        probabilities = (D_squared / np.sum(D_squared)).flatten()
        indices = np.arange(probabilities.size)
        indices_centers[centers_chosen] = np.random.choice(indices, p=probabilities)
        centers[centers_chosen] = remaining_data[indices_centers[centers_chosen]]
        centers_chosen += 1

    return centers


def silhouette(X, clusterings, k, metric=euclid):
    """
    Given results from clustering with K-means, return the silhouette measure of
    the clustering.
    :param X: The NxD data matrix.
    :param clusterings: A list of Nxk dimensional vectors, each representing the
                clustering of one of the iterations of K-means.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :return: The Silhouette statistic, for k selection. Return average of silhouette's over iterations.
    """
    # remove clustering which has empty clusters, i.e. less than k clusters
    good_clusterings = []
    for cluster in clusterings:
        if cluster.shape[1] == k:
            good_clusterings.append(cluster)
    clusterings = good_clusterings

    stats = np.zeros(len(clusterings))

    for iter in range(len(clusterings)):

        N, k = clusterings[iter].shape

        a_i = np.zeros(N)
        b_i = np.zeros(N)
        total_cluster_size_seen = 0

        # compute using matrix operations per each cluster group
        for i in range(k):

            # get all points in the cluster
            cluster_i_pts_indices = np.where(clusterings[iter][:, i] == 1)[0]
            points_in_cluster_i = X[cluster_i_pts_indices]  # cluster_size x d

            # compute distances of pairs in cluster, sum
            distances_ai = metric(points_in_cluster_i, points_in_cluster_i)  # cluster_size x cluster_size
            total_dist_ai = np.sum(distances_ai, axis=1)
            cluster_size = points_in_cluster_i.shape[0]
            a_i[total_cluster_size_seen:total_cluster_size_seen + cluster_size] = total_dist_ai / cluster_size  # cluster_size x 1

            # find minimal sum of distances from each point to all points in closest cluster
            other_clusters_dists = np.zeros((k-1, cluster_size))
            skipped_self = 0
            for j in range(k):
                if j == i:
                    skipped_self = 1
                else:
                    # fix indices in case j==i was skipped
                    j_idx = j - skipped_self

                    # get all points in the cluster j
                    cluster_j_pts_indices = np.where(clusterings[iter][:, j_idx] == 1)[0]
                    points_in_cluster_j = X[cluster_j_pts_indices]

                    # compute distances between points in clusters and points
                    distances_bi = metric(points_in_cluster_i, points_in_cluster_j)  # cluster_i_size x cluster_j_size
                    other_clusters_dists[j_idx] = np.sum(distances_bi, axis=1) / distances_bi.shape[1]  # cluster_i_size

            b_i[total_cluster_size_seen:total_cluster_size_seen + cluster_size] = np.min(other_clusters_dists, axis=0)
            total_cluster_size_seen += cluster_size

        S = (b_i - a_i) / np.maximum(a_i, b_i)
        S[np.isnan(S)] = 0
        stats[iter] = np.sum(S)

    return np.average(stats)


def kmeans(X, k, iterations=1, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init, stat=silhouette):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :param stat: A function for calculating the statistics we want to extract about
                the result (for K selection, for example).
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    statistics - whatever data you choose to use for your statistics (silhouette by default).
    """
    N, d = X.shape

    # store as lists because k might change if empty cluster is found
    final_clusters = []
    final_centroids = []

    for j in range(iterations):

        centroids = init(X, k, metric)  # kxD
        prior_centroids = np.zeros(centroids.shape)  # kxD
        while (centroids != prior_centroids).any():
            prior_centroids = centroids
            clusters = np.zeros((N, k))  # Nxk sparse matrix with 1 in (i,j) if j'th data point is in the i'th cluster

            # calculate distance of each data point from the centroids, and find closest centroids
            distance_from_centroids = metric(X, centroids)  # Nxk
            closest = np.argmin(distance_from_centroids, axis=1)  # Nx1
            clusters[np.arange(N), closest] = 1

            # calculate means of of new clusters, and set as new centroids
            for i in range(k):
                cluster_i = X[np.where(clusters[:, i] == 1)]  # cluster_size x D
                centroids[i] = center(cluster_i)

            # remove empty clusters
            good_k = np.ones(k)  # good_k[i]=1 if cluster_i is not empty
            for i in range(k):
                if np.isnan(centroids[i]).any():
                    good_k[i] = 0
            centroids = centroids[np.where(good_k == 1)]
            clusters = clusters[:, np.where(good_k == 1)].reshape(N, np.count_nonzero(good_k))
            prior_centroids = prior_centroids[np.where(good_k == 1)]

        final_clusters.append(clusters)
        final_centroids.append(centroids)

    # get statistics for all found clusters
    stats = stat(X, final_clusters, k, metric)

    # choose best clustering with minimal cost - sum of squared distances from points to their respective centroids
    costs = np.zeros(iterations)
    for j in range(iterations):

        # distance of each data point from all centroids
        distance_from_centroids = metric(X, final_centroids[j])  # Nxk

        iter_k = final_centroids[j].shape[0]
        cost_per_cluster = np.zeros(iter_k)

        for i in range(iter_k):
            points_in_cluster = np.where(final_clusters[j][:, i] == 1)[0]
            distances_per_cluster = distance_from_centroids[points_in_cluster, i]  # size_of_cluster x 1
            cost_per_cluster[i] = np.sum(distances_per_cluster)

        costs[j] = np.sum(cost_per_cluster)

    best_iter = np.argmin(costs)
    best_clustering = final_clusters[best_iter]
    best_centroids = final_centroids[best_iter]
    return best_clustering, best_centroids, stats, np.min(costs)


def heat(X, sigma):
    """
    calculate the heat kernel similarity of the given data matrix.
    :param X: A NxD data matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """
    distances = (euclid(X, X) ** 2)  # NxN
    exponent = - distances / (2 * (sigma ** 2))
    return np.exp(exponent)


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given data matrix.
    :param X: A NxD data matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    distances = euclid(X, X)  # symmetric NxN
    sorted_distances = np.argsort(distances, axis=1)[:, :m]  # N x m
    N, m = sorted_distances.shape
    cols = sorted_distances.flatten()  # vector of length Nxm
    rows = np.meshgrid(np.arange(m), np.arange(N))[1].flatten()  # vector of length mxN
    similarity = np.zeros(distances.shape)
    similarity[rows, cols] = 1
    return similarity


def spectral(X, k, similarity_param, similarity=heat, iterations=5):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the hear kernel.
    :param similarity: The similarity transformation of the data.
    :param iterations: number of iterations for kmeans algorithm.
    :return: a tuple of (clustering, centroids, statistics)
            clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
            centroids - The kxD centroid matrix.
            statistics - whatever data you choose to use for your statistics (silhouette by default).    
    """
    W = similarity(X, similarity_param)  # NxN
    D = np.diag(np.sum(W, axis=1))
    D_inv = np.linalg.pinv(np.sqrt(D))
    L = np.identity(D.shape[0]) - D_inv.dot(W.dot(D_inv))
    eigvals, eigvecs = np.linalg.eigh(L)  # returns sorted

    # choose k according to eigen values
    # plt.figure()
    # plt.plot(eigvals)
    # plt.show()

    # if any negative eigen values, take abs value and multiply matching vector by -1
    eigen_values_matrix = np.diag(eigvals)
    eigvecs[:, np.where(eigen_values_matrix < 0)[1]] *= -1
    eigvals = np.abs(eigvals)

    smallest_eigvals_indices = np.argsort(eigvals)[:k]
    lowest_eigvecs = eigvecs[:, smallest_eigvals_indices]  # column are eigenvectors, Nxk
    return kmeans(lowest_eigvecs, k, iterations)


def get_data_from_clusters(X, clusters):
    '''
    Separate data into clusters.
    :param X: A NxD data matrix.
    :param clusters: Nxk sparse matrix with 1 in (i,j) if j'th data point is in the i'th cluster
    :return: X data sorted into clusters: list of arrays each of shape (cluster_size, d)
    '''
    clustered_data = []
    for i in range(clusters.shape[1]):
        clustered_data.append(X[np.where(clusters[:, i] == 1)])
    return clustered_data


def plot_k_evaluations(type_stat, stats, k_values, data_type, spectral='Classical', similarity='', sim_param='', iterations=''):
    '''
    Plot statistics per value of k.
    :param type_stat: Name of type of statistic
    :param stats: list of statistics matching value of k in k_values
    :param list of values of k used for stats
    '''
    plt.figure()
    title = '{}: {} with {} iterations.'.format(type_stat, spectral, iterations)
    file_name = 'plots/{}/{}_{}__iters{}'.format(data_type, spectral, type_stat, iterations)
    if spectral == 'Spectral':
        title += '\nSimilarity measure: {} with parameter {}'.format(similarity, sim_param)
        file_name += '_{}_{}'.format(similarity, sim_param)
    file_name += '.png'
    plt.title(title)
    plt.scatter(k_values, stats)
    # plt.savefig(file_name)
    # plt.show()


def plot_cost(stats, k_values, data_type, spectral='Classical', similarity='', sim_param='', iterations=''):
    plot_k_evaluations('Cost', stats, k_values, data_type, spectral, similarity, sim_param, iterations)


def plot_silhoutte(stats, k_values, data_type, spectral='Classical', similarity='', sim_param='', iterations=''):
    '''
    Plot silhoutte statistics per value of k.
    :param stats: list of statistics matching value of k in k_values
    :param list of values of k used for stats
    '''
    plot_k_evaluations('Silhouette', stats, k_values, data_type, spectral, similarity, sim_param, iterations)


def plot_single_micro_cluster(data, title, filename):
    plt.figure()
    plt.title(title)
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    # plt.savefig(filename)
    # plt.show()


def plot_clusters(clustered_data, iterations, data_type='', spectral='Classical', similarity='', sim_param='',
                  clustered_images=None):

    colors = ['blue', 'plum', 'green', 'aqua', 'gray', 'orange', 'red', 'purple', 'pink', 'brown']

    k = len(clustered_data)
    fig = plt.figure()
    title = '{} Clusters with k={} and {} iterations.'.format(spectral, k, iterations)
    base_filename = 'plots/{}/{}_k{}_iters{}'.format(data_type, spectral, k, iterations)
    if spectral == 'Spectral':
        title += '\nSimilarity measure: {} with parameter {}'.format(similarity, sim_param)
        base_filename += '_{}_{}'.format(similarity, sim_param)

    filename = base_filename + '.png'
    plt.suptitle(title)
    for i in range(k):
        if data_type == 'microarray':
            plot_single_micro_cluster(clustered_data[i], title + '\nCluster %d' % i, base_filename + '_cluster%d.png' % i)
        elif data_type == 'MNIST':

            for j in range(0, min(clustered_data[i].shape[0], clustered_images[i].shape[0]), 5):
                plt.text(clustered_data[i][j, 0], clustered_data[i][j, 1], str(clustered_images[i][j][0]), color=colors[
                    clustered_images[i][j]])
                # invisible plot in order to center figure around data
                plt.plot(clustered_data[i][j, 0], clustered_data[i][j, 1], alpha=0)
        else:
            x_vals = clustered_data[i][:, 0]
            y_vals = clustered_data[i][:, 1]
            plt.scatter(x_vals, y_vals, cmap=plt.cm.rainbow)
            if data_type == 'Pac Man':
                fig = plot_with_images(clustered_data[i], clustered_images[i], '', fig)
    # fig.savefig(filename)
    # plt.show()


def reduce_biological_data():
    data = microarray_exploration()
    reduced_data = TSNE(n_components=2).fit_transform(data)
    return reduced_data


def plot_with_images(X, images, title, figure, image_num=10):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param figure: plt.figure()
    :param image_num: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''
    n, pixels = np.shape(images)
    n = min(n, X.shape[0])
    img_size = int(pixels**0.5)

    fig = figure
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


def pacman_reduced_data():
    img_data = imread('Pac Man.jpg', mode='L')

    rotated_images = np.array([copy.deepcopy(img_data)]).flatten().reshape(1, img_data.size)
    for angle in range(0, 360, 1):
        rotated = np.array([rotate(img_data, angle, reshape=False)]).flatten().reshape(1, img_data.size)
        rotated_images = np.concatenate((rotated_images, rotated), axis=0)

    reduced_data = TSNE(n_components=2).fit_transform(rotated_images)
    return reduced_data, rotated_images


def mnist_reduced_data():
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target
    reduced_data = TSNE(n_components=2).fit_transform(data)
    return reduced_data, labels


if __name__ == '__main__':
    k = 8
    iters = 20
    images = None

    # UNCOMMENT THE DATA TYPE
    # data, data_name = np.transpose(circles_example()), 'circles'
    data, data_name = (apml_pic_example('APML_pic.pickle')), 'apml2017'
    # data, data_name = reduce_biological_data(), 'microarray'
    # (data, images), data_name = pacman_reduced_data(), 'Pac Man'
    # (data, images), data_name = mnist_reduced_data(), 'MNIST'
    # images = np.array(images).reshape(len(images), 1)

    all_stats = []
    all_spec_stats = []
    all_costs = []
    all_spec_costs = []
    real_kvals = []
    real_spectral_kvals = []

    # CHOOSE SIMILARITY FUNCTION AND PARAMETER
    sim_func = mnn
    sim_name = 'mnn'
    sim_param = 20

    for kk in range(3, k):
        clusters, centroids, stats, costs = kmeans(data, kk, iters)
        spec_clusters, spec_centroids, spec_stats, spec_costs = spectral(data, kk, sim_param, sim_func, iters)

        real_kvals.append(len(centroids))
        real_spectral_kvals.append(len(spec_centroids))

        all_stats.append(stats)
        all_spec_stats.append(spec_stats)
        all_costs.append(costs)
        all_spec_costs.append(spec_costs)

        clustered_images = None
        if data_name == 'Pac Man' or data_name == 'MNIST':
            clustered_images = get_data_from_clusters(images, clusters)

        plot_clusters(get_data_from_clusters(data, clusters), str(iters), data_name, clustered_images=clustered_images)
        plot_clusters(get_data_from_clusters(data, spec_clusters), str(iters), data_name, 'Spectral', sim_name,
                      str(sim_param), clustered_images)

    # Plot silhouette and costs
    plot_silhoutte(all_stats, real_kvals, data_name, iterations=str(iters))
    plot_silhoutte(all_spec_stats, real_spectral_kvals, data_name, 'Spectral', sim_name, str(sim_param), str(iters))
    plot_cost(all_costs, real_kvals, data_name, iterations=str(iters))
    plot_cost(all_spec_costs, real_spectral_kvals, data_name, 'Spectral', sim_name, str(sim_param), str(iters))
