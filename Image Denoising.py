import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
import pickle
from skimage.util import view_as_windows as viewW


def images_example(path='train_images.pickle'):
    """
    A function demonstrating how to access to image data supplied in this exercise.
    :param path: The path to the pickle file.
    """
    patch_size = (8, 8)

    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    patches = sample_patches(train_pictures, psize=patch_size, n=20000)

    plt.figure()
    plt.imshow(train_pictures[0])
    plt.title("Picture Example")

    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(patches[:,i].reshape(patch_size), cmap='gray')
        plt.title("Patch Example")
    plt.show()


def im2col(A, window, stepsize=1):
    """
    an im2col function, transferring an image to patches of size window (length
    2 list). the step size is the stride of the sliding window.
    :param A: The original image (NxM size matrix of pixel values).
    :param window: Length 2 list of 2D window size.
    :param stepsize: The step size for choosing patches (default is 1).
    :return: A (heightXwidth)x(NxM) matrix of image patches.
    """
    return viewW(np.ascontiguousarray(A), (window[0], window[1])).reshape(-1,
                        window[0] * window[1]).T[:, ::stepsize]


def greyscale_and_standardize(images, remove_mean=True):
    """
    The function receives a list of RGB images and returns the images after
    grayscale, centering (mean 0) and scaling (between -0.5 and 0.5).
    :param images: A list of images before standardisation.
    :param remove_mean: Whether or not to remove the mean (default is True).
    :return: A list of images after standardisation.
    """
    standard_images = []

    for image in images:
        standard_images.append((0.299 * image[:, :, 0] +
                                0.587 * image[:, :, 1] +
                                0.114 * image[:, :, 2]) / 255)

    sum = 0
    pixels = 0
    for image in standard_images:
        sum += np.sum(image)
        pixels += image.shape[0] * image.shape[1]
    dataset_mean_pixel = float(sum) / pixels

    if remove_mean:
        for image in standard_images:
            image -= np.matlib.repmat([dataset_mean_pixel], image.shape[0],
                                      image.shape[1])

    return standard_images


def sample_patches(images, psize=(8, 8), n=10000, remove_mean=True):
    """
    sample N p-sized patches from images after standardising them.

    :param images: a list of pictures (not standardised).
    :param psize: a tuple containing the size of the patches (default is 8x8).
    :param n: number of patches (default is 10000).
    :param remove_mean: whether the mean should be removed (default is True).
    :return: A matrix of n patches from the given images.
    """
    d = psize[0] * psize[1]
    patches = np.zeros((d, n))
    standardized = greyscale_and_standardize(images, remove_mean)

    shapes = []
    for pic in standardized:
        shapes.append(pic.shape)

    rand_pic_num = np.random.randint(0, len(standardized), n)
    rand_x = np.random.rand(n)
    rand_y = np.random.rand(n)

    for i in range(n):
        pic_id = rand_pic_num[i]
        pic_shape = shapes[pic_id]
        x = int(np.ceil(rand_x[i] * (pic_shape[0] - psize[1])))
        y = int(np.ceil(rand_y[i] * (pic_shape[1] - psize[0])))
        patches[:, i] = np.reshape(np.ascontiguousarray(
            standardized[pic_id][x:x + psize[0], y:y + psize[1]]), d)

    return patches


def denoise_image(Y, model, denoise_function, noise_std, patch_size=(8, 8)):
    """
    A function for denoising an image. The function accepts a noisy gray scale
    image, denoises the different patches of it and then reconstructs the image.

    :param Y: the noisy image.
    :param model: a Model object (MVN/ICA/GSM).
    :param denoise_function: a pointer to one of the denoising functions (that corresponds to the model).
    :param noise_std: the noise standard deviation parameter.
    :param patch_size: the size of the patch that the model was trained on (default is 8x8).
    :return: the denoised image, after each patch was denoised. Note, the denoised image is a bit
    smaller than the original one, since we lose the edges when we look at all of the patches
    (this happens during the im2col function).
    """
    (h, w) = np.shape(Y)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))

    # split the image into columns and denoise the columns:
    noisy_patches = im2col(Y, patch_size)
    denoised_patches = denoise_function(noisy_patches, model, noise_std)

    # reshape the denoised columns into a picture:
    x_hat = np.reshape(denoised_patches[middle_linear_index, :],
                       [cropped_h, cropped_w])

    return x_hat


def crop_image(X, patch_size=(8, 8)):
    """
    crop the original image to fit the size of the denoised image.
    :param X: The original picture.
    :param patch_size: The patch size used in the model, to know how much we need to crop.
    :return: The cropped image.
    """
    (h, w) = np.shape(X)
    cropped_h = h - patch_size[0] + 1
    cropped_w = w - patch_size[1] + 1
    middle_linear_index = int(
        ((patch_size[0] / 2) * patch_size[1]) + (patch_size[1] / 2))
    columns = im2col(X, patch_size)
    return np.reshape(columns[middle_linear_index, :], [cropped_h, cropped_w])


def normalize_log_likelihoods(X):
    """
    Given a matrix in log space, return the matrix with normalized columns in
    log space.
    :param X: Matrix in log space to be normalised.
    :return: The matrix after normalization.
    """
    h, w = np.shape(X)
    return X - np.matlib.repmat(logsumexp(X, axis=0), h, 1)


def test_denoising(image, model, denoise_function,
                   noise_range=(0.01, 0.05, 0.1, 0.2), patch_size=(8, 8)):
    """
    A simple function for testing your denoising code. You can and should
    implement additional tests for your code.
    :param image: An image matrix.
    :param model: A trained model (MVN/ICA/GSM).
    :param denoise_function: The denoise function that corresponds to your model.
    :param noise_range: A tuple containing different noise parameters you wish
            to test your code on. default is (0.01, 0.05, 0.1, 0.2).
    :param patch_size: The size of the patches you've used in your model.
            Default is (8, 8).
    """
    h, w = np.shape(image)
    noisy_images = np.zeros((h, w, len(noise_range)))
    denoised_images = []
    cropped_original = crop_image(image, patch_size)

    # make the image noisy:
    for i in range(len(noise_range)):
        noisy_images[:, :, i] = image + (
        noise_range[i] * np.random.randn(h, w))

    # denoise the image:
    for i in range(len(noise_range)):
        denoised_images.append(
            denoise_image(noisy_images[:, :, i], model, denoise_function,
                          noise_range[i], patch_size))

    # calculate the MSE for each noise range:
    for i in range(len(noise_range)):
        print("noisy MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((crop_image(noisy_images[:, :, i],
                                  patch_size) - cropped_original) ** 2))
        print("denoised MSE for noise = " + str(noise_range[i]) + ":")
        print(np.mean((cropped_original - denoised_images[i]) ** 2))

    plt.figure()
    for i in range(len(noise_range)):
        plt.subplot(2, len(noise_range), i + 1)
        plt.imshow(noisy_images[:, :, i], cmap='gray')
        plt.subplot(2, len(noise_range), i + 1 + len(noise_range))
        plt.imshow(denoised_images[i], cmap='gray')
    plt.show()


class GMM_Model:
    """
    A class that represents a Gaussian Mixture Model, with all the parameters
    needed to specify the model.

    mix - a length k vector with the multinomial parameters for the gaussians.
    means - a k-by-D matrix with the k different mean vectors of the gaussians.
    cov - a k-by-D-by-D tensor with the k different covariance matrices.
    """
    def __init__(self, mixture, means, cov):
        self.mix = mixture
        self.means = means
        self.cov = cov


class MVN_Model:
    """
    A class that represents a Multivariate Gaussian Model, with all the parameters
    needed to specify the model.

    mean - a D sized vector with the mean of the gaussian.
    cov - a D-by-D matrix with the covariance matrix.
    gmm - a GMM_Model object.
    """
    def __init__(self, mean, cov, gmm):
        self.mean = mean
        self.cov = cov
        self.gmm = gmm


class GSM_Model:
    """
    A class that represents a GSM Model, with all the parameters needed to specify
    the model.

    cov - a k-by-D-by-D tensor with the k different covariance matrices. the
        covariance matrices should be scaled versions of each other.
    mix - k-length probability vector for the mixture of the gaussians.
    gmm - a GMM_Model object.
    """
    def __init__(self, cov, mix, gmm):
        self.cov = cov
        self.mix = mix
        self.gmm = gmm


class ICA_Model:
    """
    A class that represents an ICA Model, with all the parameters needed to specify
    the model.

    P - linear transformation of the sources. (X = P*S)
    vars - DxK matrix whose (d,k) element corresponds to the variance of the k'th
        gaussian in the d'th source.
    mix - DxK matrix whose (d,k) element corresponds to the weight of the k'th
        gaussian in d'th source.
    gmms - A list of K GMM_Models, one for each source.
    """
    def __init__(self, P, vars, mix, gmms):
        self.P = P
        self.vars = vars
        self.mix = mix
        self.gmms = gmms


def MVN_log_likelihood(X, model):
    """
    Given image patches and a MVN model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_size X number_of_patches matrix of image patches.
    :param model: A MVN_Model object.
    :return: The log likelihood of all the patches combined.
    """
    return np.sum(np.log(multivariate_normal.pdf(X, model.mean, model.cov, allow_singular=True)))


def GSM_log_likelihood(X, model):
    """
    Given image patches and a GSM model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: A GSM_Model object.
    :return: The log likelihood of all the patches combined.
    """

    inners = np.zeros((N,k))
    for kk in range(k):
        inners[:, kk] = model.mix[kk] * multivariate_normal.pdf(X, model.gmm.means[kk], model.cov[kk], allow_singular=True)
    return np.sum(np.log(np.sum(inners, axis=1)))


def ICA_log_likelihood(X, model):
    """
    Given image patches and an ICA model, return the log likelihood of the patches
    according to the model.

    :param X: a patch_sizeXnumber_of_patches matrix of image patches.
    :param model: An ICA_Model object.
    :return: The log likelihood of all the patches combined.
    """

    # TODO: YOUR CODE HERE


def learn_GMM(X, k, initial_model, learn_mixture=True, learn_means=True,
              learn_covariances=True, iterations=10):
    """
    A general function for learning a GMM_Model using the EM algorithm.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: number of components in the mixture.
    :param initial_model: an initial GMM_Model object to initialize EM with.
    :param learn_mixture: a boolean for whether or not to learn the mixtures.
    :param learn_means: a boolean for whether or not to learn the means.
    :param learn_covariances: a boolean for whether or not to learn the covariances.
    :param iterations: Number of EM iterations (default is 10).
    :return: (GMM_Model, log_likelihood_history)
            GMM_Model - The learned GMM Model.
            log_likelihood_history - The log-likelihood history for debugging.
    """
    X = X.transpose()
    learnt_mean, learnt_cov, learnt_mix, lle = Expectation_Maximization(X, k, initial_model, iterations)
    model = GMM_Model(learnt_mix, learnt_mean, learnt_cov)
    return model, lle


def learn_MVN(X, k):
    """
    Learn a multivariate normal model, given a matrix of image patches.
    :param X: a dxN data matrix, where d is the dimension, and N is the number of samples.
    :param k: number of guassians
    :return: A trained MVN_Model object.
    """
    X = X.T
    N, d = X.shape
    mean = np.mean(X, axis=0)
    cov_tmp = X - mean[None,:]
    cov = cov_tmp.T.dot(cov_tmp) / N

    return MVN_Model(mean, cov, None)


def learn_GSM(X, k):
    """
    Learn parameters for a Gaussian Scaling Mixture model for X using EM.

    GSM components share the variance, up to a scaling factor, so we only
    need to learn scaling factors and mixture proportions.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components of the GSM model.
    :return: A trained GSM_Model object.
    """
    X = X.transpose()

    #initialize model
    mean = np.zeros((k, 64))
    cov_tmp = X - mean[0]
    single_cov = cov_tmp.T.dot(cov_tmp) / N
    base_cov = np.array([single_cov] * k)
    mix = np.array([float(1 / k)] * k)
    gmm_model = GMM_Model(mix, mean, base_cov)
    initial_model = GSM_Model(base_cov, mix, gmm_model)

    learnt_mean, learnt_cov, learnt_mix, lle = Expectation_Maximization(X, k, initial_model, max_iterations=5, learn_gsm=True)
    learnt_model = GSM_Model(learnt_cov, learnt_mix, GMM_Model(learnt_mix, learnt_mean, learnt_cov))
    return learnt_model


def learn_ICA(X, k):
    """
    Learn parameters for a complete invertible ICA model.

    We learn a matrix P such that X = P*S, where S are D independent sources
    And for each of the D coordinates we learn a mixture of K univariate
    0-mean gaussians using EM.

    :param X: a DxM data matrix, where D is the dimension, and M is the number of samples.
    :param k: The number of components in the source gaussian mixtures.
    :return: A trained ICA_Model object.
    """
    d, N = X.shape
    cov = (X.dot(X.T)) / N
    eigvecs, _ = np.linalg.eigh(cov)
    S = eigvecs.T.dot(X)
    vars = np.zeros((d, k))
    mixtures = np.zeros((d, k))

    gmms = []
    for i in range(d):
        single = S[i:,]
        mix = np.array([float(1 / k)] * k)
        means = np.zeros((k, 1))
        var = np.array([single.T.dot(single)] * k).reshape(k, 1, 1) / N
        initial_model = GMM_Model(mix, means, var)
        model, lle = learn_GMM(single[None,:], k, initial_model, iterations=5)
        gmms.append(model)
        vars[i] = model.cov[:,0,0]
        mixtures[i] = model.mix

    learnt_ica_model = ICA_Model(eigvecs, vars, mixtures, gmms)
    return learnt_ica_model


def Expectation_Maximization(samples, k, model, max_iterations=20, learn_gsm=False):
    '''
    Run the iterative EM Algorithm on the given data.
    :param samples: [numpy.ndarray] 2d array, each row contains a flattened image patch of size d. Nxd
    :param k: number of guassians
    :param max_iterations: stop at this number of iterations, if not yet converged
    :param learn_gsm: if True, do not learn mean and assume cov=r*single_cov, where we need to learn r.
    :return: mean, covariance, pi, list of log likelihoods for each iteration
    '''
    d = samples.shape[1]
    N = samples.shape[0]

    # initialize parameters
    c = np.ones((N, k))  # probability of y given a guassian distribution, N x k
    pi = model.mix  # mixture, probability of y, multiplication of c, 1 x k
    covariance = model.cov  # k x d x d
    r_squared = np.ones((k,1)) / k  # coefficient of each covariance
    if learn_gsm:
        mean = model.gmm.means
    else:
        mean = model.means  # k x d

    # for gsm learning
    if learn_gsm:
        cov_tmp = samples - mean[0]
        single_cov = cov_tmp.T.dot(cov_tmp) / N
        base_cov = np.array([single_cov] * k)
        covariance = base_cov

    # convergence parameter
    initial_c = c + 1
    epsilon = 0.1

    loglikelihoods = []
    iter = 0

    while (np.abs(initial_c - c) > epsilon).any() and iter < max_iterations:
        for kk in range(k):
            # calculate c: calculate numerator and denominator separately in logspace,
            # and then divide and revert to original space
            c[:, kk] = np.log(pi[kk]) + multivariate_normal.logpdf(samples, mean[kk], covariance[kk], allow_singular=True)

        denominator_log = logsumexp(c, axis=1).reshape(N, 1)
        c = np.exp(c - denominator_log)

        # calculate pi
        pi = np.sum(c, axis=0) / d

        # calculate mu (mean)
        if not learn_gsm:
            for kk in range(k):
                mean[kk] = np.dot(c[:, kk].T, samples) / np.sum(c[:, kk])

        # calculate r
        if learn_gsm:
            for kk in range(k):
                numerator = np.sum(c[:, kk] * np.diag(samples.dot(np.linalg.pinv(base_cov[kk])).dot(samples.T)))
                r_squared[kk] = numerator / (d * np.sum(c[:, kk], axis=0))

        # calculate Sigma (covariance)
        else:
            for kk in range(k):
                if learn_gsm:
                    covariance[i] = base_cov[i] * r_squared[i]
                else:
                    covariance[kk] = np.dot(c[:, kk] * (samples - mean[kk]).T, (samples - mean[kk]))
                    covariance[kk] = covariance[kk] / np.sum(c[:, kk])

        # calculate log likelihood of parameters
        inner_likelihood_matrix = np.array(c)
        for kk in range(k):
            inner = pi[kk] * multivariate_normal.pdf(samples, mean[kk], covariance[kk], allow_singular=True)
            inner_likelihood_matrix[:, kk] = c[:, kk] * np.log(inner)

        likelihood = 1 + np.sum(np.sum(inner_likelihood_matrix, axis=1), axis=0)
        loglikelihoods.append(likelihood)

        iter += 1

    return mean, covariance, pi, loglikelihoods


def weiner_filter(y, mean, cov, noise_std):
    '''
    
    :param y: samples N x d
    :param mean: array k x d
    :param cov: array k x d x d
    :param noise_std: scalar
    :return: array N x d
    '''
    d = y.shape[1]
    cov_inv = np.linalg.pinv(cov)
    weiner = np.linalg.pinv(cov_inv + (np.identity(d) / np.power(noise_std, 2)))
    weiner = (cov_inv.dot(mean) + (y / np.power(noise_std, 2))).dot(weiner)
    return weiner


def MVN_Denoise(Y, mvn_model, noise_std):
    """
    Denoise every column in Y, assuming an MVN model and gaussian white noise.
    The model assumes that y = x + noise where x is generated by a single
    0-mean multi-variate normal distribution.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param mvn_model: The MVN_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    Y = Y.transpose()
    X_estimate = weiner_filter(Y, mvn_model.mean, mvn_model.cov, noise_std)
    return X_estimate.transpose()


def GMM_Denoise(X, gmm_model, noise_std):
    """
    Denoise every column in X, assuming a GMM model and gaussian white noise.
    :param X: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gmm: The GMM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    X = X.transpose()
    N, d = X.shape
    c_log = np.zeros((N, k))
    weiners = np.zeros((k, N, d))

    for kk in range(k):
        cov = gmm_model.cov[kk]
        mix = gmm_model.mix[kk]
        mean = gmm_model.means[kk]
        weiners[kk] = weiner_filter(X, mean, cov, noise_std)

        # calculate posterior probability
        c_log[:, kk] = np.log(mix) + multivariate_normal.logpdf(X, mean, cov, allow_singular=True)

    denominator_log = logsumexp(c_log, axis=1)[:, None]
    c = np.exp(c_log - denominator_log)

    for kk in range(k):
        weiners[kk] = c[:, kk][:, None] * weiners[kk]
    reconstruction = np.sum(weiners, axis=0)
    return reconstruction.transpose()



def GSM_Denoise(Y, gsm_model, noise_std):
    """
    Denoise every column in Y, assuming a GSM model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by a mixture of
    0-mean gaussian components sharing the same covariance up to a scaling factor.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param gsm_model: The GSM_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.

    """
    X = Y.transpose()
    N, d = X.shape
    c_log = np.zeros((N, k))
    weiners = np.zeros((k, N, d))

    for kk in range(k):
        cov = gsm_model.cov[kk]
        mix = gsm_model.mix[kk]
        mean = gsm_model.gmm.means[kk]
        weiners[kk] = weiner_filter(X, mean, cov, noise_std)

        # calculate posterior probability
        c_log[:, kk] = np.log(mix) + multivariate_normal.logpdf(X, mean, cov, allow_singular=True)

    denominator_log = logsumexp(c_log, axis=1)[:, None]
    c = np.exp(c_log - denominator_log)

    for kk in range(k):
        weiners[kk] = c[:, kk][:, None] * weiners[kk]
    reconstruction = np.sum(weiners, axis=0)
    return reconstruction.transpose()


def ICA_Denoise(Y, ica_model, noise_std):
    """
    Denoise every column in Y, assuming an ICA model and gaussian white noise.

    The model assumes that y = x + noise where x is generated by an ICA 0-mean
    mixture model.

    :param Y: a DxM data matrix, where D is the dimension, and M is the number of noisy samples.
    :param ica_model: The ICA_Model object.
    :param noise_std: The standard deviation of the noise.
    :return: a DxM matrix of denoised image patches.
    """
    Y = Y.T
    N, d = Y.shape
    cov = (Y.T.dot(Y)) / N
    _, eigvecs = np.linalg.eigh(cov)
    S = eigvecs.T.dot(Y.T)
    reconstruction = np.zeros(S.shape)

    for j in range(d):
        single = S[j, :][:,None].T
        gmm = ica_model.gmms[j]
        reconstruction[j] = GMM_Denoise(single, gmm, noise_std)

    final = eigvecs.dot(reconstruction)
    return final


if __name__ == '__main__':

    patch_size = (8, 8)
    with open('train_images.pickle', 'rb') as f:
        train_pictures = pickle.load(f)

    N = 2000
    patches = sample_patches(train_pictures, psize=patch_size, n=N)
    k = 3

    # model, denoise_func = learn_MVN(patches, 1), ICA_Denoise
    model, denoise_func = learn_GSM(patches, k), GSM_Denoise
    # model, denoise_func = learn_ICA(patches, k), GMM_Denoise

    img = (greyscale_and_standardize(train_pictures)[0])
    test_denoising(img, model, denoise_func, patch_size=patch_size)
