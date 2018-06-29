import pickle
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# train network on MNIST data
from tensorflow.examples.tutorials.mnist import input_data

class AutoEncoderNetwork:
    '''
    AutoEncoderNetwork class encompasses a neural network which implements an auto-encoder with two hidden layers.
    '''

    ENCODER = 'encoder'
    DECODER = 'decoder'

    def __init__(self, data, input_size, hidden1_input_size, hidden2_input_size, batch_size, num_epochs, learning_rate):
        '''
        :param data: []
        :param input_size: [int] size of input to network
        :param hidden1_input_size: [int] size of input to the first hidden layer.
        :param hidden2_input_size: [int] size of input to the second hidden layer.
        :param batch_size: [int]
        :param num_epochs: [int]
        :param learning_rate: [float]
        '''
        self._session = None
        self._mnist_data = data
        self.input_size = input_size
        self._h1_input_size = 784
        self._h2_input_size = 256
        self._h3_input_size = 64
        self._h4_input_size = 32
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._batches_per_epoch = int(np.floor(self._mnist_data.train.num_examples / batch_size))
        self._learning_rate = learning_rate


        # hold graph input for images
        # self._placeholder_input = tf.placeholder(tf.float32, (None, 32, 32, 1))
        self._placeholder_input = tf.placeholder("float", [None, self.input_size])
        self._placeholder_input_latent = tf.placeholder("float", [None, 32])

        self._encoder = self.encoder_layer(self._placeholder_input)
        self._decoder = self.decoder_layer(self._encoder)

        # use squared error as loss function
        self._loss = tf.reduce_mean(tf.pow(self._placeholder_input - self._decoder, 2))

        # optimizing the network
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._loss)

        # initialize variables
        try:
            self._initial_vars = tf.initialize_all_variables()
        except AttributeError:
            # using a different version of tensorflow
            self._initial_vars = tf.global_variables_initializer()

        # create tensorflow session
        self._session = tf.Session()
        self._session.run(self._initial_vars)

        self._trained = False
        self._cost_epochs = []
        self._costs = []


    def train(self, slower_learning_rate, epoch_switch_rate):
        '''
        Train network.
        :param slower_learning_rate: [float]
        :param epoch_switch_rate
        '''
        if not self._session:
            raise ValueError("Session has not been started yet.")

        reduced_rate = False

        for epoch in range(self._num_epochs):
            if epoch % 100 == 0:
                print("Epoch %d" % epoch)
            img, label = self._mnist_data.train.next_batch(self._batch_size)
            _, cost = self._session.run([self._optimizer, self._loss], feed_dict={self._placeholder_input: img})

            # reduce learning rate after a fraction of the iterations
            if cost < 0.038 and not reduced_rate:
                self._learning_rate = slower_learning_rate

            # collect costs for analysis
            # if batch == self._batches_per_epoch - 1:
            #     print('Epoch: {}, Cost={:.5f}'.format((epoch + 1), cost))
            #     self._cost_epochs.append(epoch)
            #     self._costs.append(cost)

        self._trained = True
        self.plot_cost()

    def plot_cost(self):
        plt.figure()
        plt.plot(self._cost_epochs, self._costs)
        plt.title("Learning with {} epochs, batch size{} and learning rate {}".format(self._num_epochs,
                                                                                         self._batch_size, self._learning_rate))

    def test(self, plot=False):
        '''
        Test trained network.
        :param plot: [bool] plot reconstructed image with its original image
        :return: error [float]
        '''
        if not self._trained:
            raise ValueError("Network has not yet been trained.")
        test_img, test_label = self._mnist_data.test.next_batch(50)
        reconstructed_img, cost = self._session.run([self._decoder, self._loss], feed_dict={self._placeholder_input: test_img})

        if plot:
            self.plot_compare_reconstruction(test_img, reconstructed_img)

        return cost

    def restore_dim(self, reduced):
        reduced = np.array(reduced).astype(np.float32)
        orig_imgs = self._session.run(self._decoder, feed_dict={self._encoder: reduced})
        plt.figure()
        img = orig_imgs[3].reshape((28, 28))
        plt.imshow(img)
        plt.show()

    def reduce_dim(self):
        # if not self._session or not self._trained:
        #     raise ValueError("Network has not yet been trained.")

        imgs, labels = self._mnist_data.test.next_batch(256)
        low_dim_img = self._session.run([self._encoder], feed_dict={self._placeholder_input: imgs})[0]
        title = "Embedding (learning parameters: {} epochs, {} batch size and {} learning rate".format(
            self._num_epochs, self._batch_size, self._learning_rate)
        self.display_mnist_digits(low_dim_img, labels, title)
        return low_dim_img

    @staticmethod
    def display_mnist_digits(digits, labels, title):
        '''
        Perform different manifold dim reduction algorithms on MNIST digits data. Plot results.
        :param func: function to perform on data, implementing algorithm. One of: MDS, LLE, DiffusionMap
        :param kwargs: args to pass to func
        '''
        colors = ['blue', 'plum', 'green', 'aqua', 'gray', 'orange', 'red', 'purple', 'pink', 'brown']
        plt.figure()
        for i in range(digits.shape[0]):
            label = np.where(labels[i] == 1)[0][0]
            plt.text(digits[i, 0], digits[i, 1], str(label), color=colors[label])
            # invisible plot in order to center figure around data
            plt.plot(digits[i, 0], digits[i, 1], alpha=0)
            plt.axis('off')
        plt.title(title)
        plt.show()

    def weights(self, layer_type, layer_number):
        '''
        :param layer_type: [str] encoder or decoder
        :param layer_number: [int] first or second
        :return: [tf.Variable] of a random normal variable
        '''
        var_shape = None
        if layer_type == self.ENCODER:
            if layer_number == 1:
                var_shape = [self.input_size, self._h1_input_size]
            elif layer_number == 2:
                var_shape = [self._h1_input_size, self._h2_input_size]
            elif layer_number == 3:
                var_shape = [self._h2_input_size, self._h3_input_size]
            elif layer_number == 4:
                var_shape = [self._h3_input_size, self._h4_input_size]
        elif layer_type == self.DECODER:
            if layer_number == 1:
                var_shape = [self._h4_input_size, self._h3_input_size]
            elif layer_number == 2:
                var_shape = [self._h3_input_size, self._h2_input_size]
            elif layer_number == 3:
                var_shape = [self._h2_input_size, self._h1_input_size]
            elif layer_number == 4:
                var_shape = [self._h1_input_size, self.input_size]
        if not var_shape:
            raise ValueError("Invalid parameters to AutoEncoderNetwork.weights")
        return tf.Variable(tf.random_normal(var_shape))

    def biases(self, layer_type, layer_number):
        '''
        :param layer_type: [str] encoder or decoder
        :param layer_number: [int] first or second
        :return: [tf.Variable] of a random normal variable
        '''
        var_shape = None
        if layer_type == self.ENCODER:
            if layer_number == 1:
                var_shape = [self._h1_input_size]
            elif layer_number == 2:
                var_shape = [self._h2_input_size]
            elif layer_number == 3:
                var_shape = [self._h3_input_size]
            elif layer_number == 4:
                var_shape = [self._h4_input_size]
        elif layer_type == self.DECODER:
            if layer_number == 1:
                var_shape = [self._h3_input_size]
            elif layer_number == 2:
                var_shape = [self._h2_input_size]
            elif layer_number == 3:
                var_shape = [self._h1_input_size]
            elif layer_number == 4:
                var_shape = [self.input_size]
        if not var_shape:
            raise ValueError("Invalid parameters to AutoEncoderNetwork.weights")
        return tf.Variable(tf.random_normal(var_shape))

    def narrow_encoder_layer(self, input):
        '''
        Build the encoder layer of the network.
        :param input: []
        :return: [] layer representing the encoder
        '''
        # multiply input with weights, and add bias
        output1 = tf.add(tf.matmul(input, self.weights(self.ENCODER, 1)), self.biases(self.ENCODER, 1))

        layer1 = tf.nn.sigmoid(output1)
        # apply activation function

        # again for second layer with layer1 as input
        output2 = tf.add(tf.matmul(layer1, self.weights(self.ENCODER, 2)), self.biases(self.ENCODER, 2))

        # apply activation function
        layer2 = tf.nn.sigmoid(output2)

        return layer2

    def encoder_layer(self, input):
        '''
        Build the encoder layer of the network.
        :param input: []
        :return: [] layer representing the encoder
        '''
        # fully connected + sigmoid activation

        output1 = tf.add(tf.matmul(input, self.weights(self.ENCODER, 1)), self.biases(self.ENCODER, 1))
        layer1 = tf.nn.sigmoid(output1)

        output2 = tf.add(tf.matmul(layer1, self.weights(self.ENCODER, 2)), self.biases(self.ENCODER, 2))
        layer2 = tf.nn.sigmoid(output2)

        outpu3 = tf.add(tf.matmul(layer2, self.weights(self.ENCODER, 3)), self.biases(self.ENCODER, 3))
        layer3 = tf.nn.sigmoid(outpu3)

        output4 = tf.add(tf.matmul(layer3, self.weights(self.ENCODER, 4)), self.biases(self.ENCODER, 4))
        layer4 = tf.nn.sigmoid(output4)

        return layer4

    def decoder_layer(self, input):
        '''
        Build the decoder layer of the network.
        :param input: []
        :return: [] layer representing the encoder
        '''
        # multiply input with weights, and add bias
        output1 = tf.add(tf.matmul(input, self.weights(self.DECODER, 1)), self.biases(self.DECODER, 1))
        layer1 = tf.nn.sigmoid(output1)

        output2 = tf.add(tf.matmul(layer1, self.weights(self.DECODER, 2)), self.biases(self.DECODER, 2))
        layer2 = tf.nn.sigmoid(output2)

        output3 = tf.add(tf.matmul(layer2, self.weights(self.DECODER, 3)), self.biases(self.DECODER, 3))
        layer3 = tf.nn.sigmoid(output3)

        output4 = tf.add(tf.matmul(layer3, self.weights(self.DECODER, 4)), self.biases(self.DECODER, 4))
        layer4 = tf.nn.sigmoid(output4)

        return layer4

    def narrow_decoder_layer(self, input):
        '''
        Build the decoder layer of the network.
        :param input: []
        :return: [] layer representing the encoder
        '''
        # multiply input with weights, and add bias
        output1 = tf.add(tf.matmul(input, self.weights(self.DECODER, 1)), self.biases(self.DECODER, 1))

        # apply activation function
        layer1 = tf.nn.sigmoid(output1)

        # again for second layer with layer1 as input
        output2 = tf.add(tf.matmul(layer1, self.weights(self.DECODER, 2)), self.biases(self.DECODER, 2))

        # apply activation function
        layer2 = tf.nn.sigmoid(output2)

        return layer2

    def latent_interpolation(self, batch):

        # take two images from batch
        img1 = batch[0, :].reshape([1, batch.shape[1]])
        img2 = batch[1, :].reshape([1, batch.shape[1]])
        self.interpolate_images(img1, img2, "Latent Space Interpolation", True)

    def __del__(self):
        # make sure tensorflow session releases all resources
        if self._session:
            self._session.close()
            self._session = None

    def __enter__(self):
        print("AutoEncoderNetwork created.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure tensorflow session releases all resources
        if self._session:
            self._session.close()
            self._session = None

    def plot_compare_reconstruction(self, test_img, reconstructed_img):
        plt.figure(1)
        plt.title('Reconstructed Images')
        for i in range(50):
            plt.subplot(5, 10, i + 1)
            img = reconstructed_img[i, :].reshape((28, 28))
            plt.imshow(img, cmap='gray')
        plt.figure(2)
        plt.title('Original Images')
        for i in range(50):
            plt.subplot(5, 10, i + 1)
            img = test_img[i, :].reshape((28, 28))
            plt.imshow(img, cmap='gray')
        plt.show()

    def pca_reduction(self, data, labels, dim, reverse=False):
        '''
        Perform and display dimension reduction on given data to dimension 2.
        :param data:
        :param labels:
        :param reverse: [bool] if true, perform reverse transformation and display reconstructed data.
        '''
        try:
            pca = PCA(n_components=dim)
            reduced = pca.fit_transform(data)
            self.interpolate_images(reduced[0], reduced[1], "PCA Space Interpolation", True)
            # AutoEncoderNetwork.display_mnist_digits(reduced, labels, "PCA dimension reduction MNIST")
            if reverse:
                original = pca.inverse_transform(reduced)
                self.plot_compare_reconstruction(data, original)
                # calculate error
                error = np.mean((data - original) ** 2)
                print(type(error))
                print("PCA reconstruction error: %.4f" % error)
        except Exception:
            # wrong version installed
            return

    def interpolate_images(self, img1, img2, title, latent=False):
        '''Interpolate and display given images.'''

        interpolated = img2.copy()
        interp_levels = 6

        # interpolation factors
        a = 0.35
        b = 0.35

        # give first image more dominance
        interpolated = np.concatenate((interpolated, (a * img1 + (1 - a) * img2)), axis=0)
        interpolated = np.concatenate((interpolated, (b * img1 + (1 - b) * img2)), axis=0)

        # give second image more dominance
        interpolated = np.concatenate((interpolated, ((1 - b) * img1 + b * img2)), axis=0)
        interpolated = np.concatenate((interpolated, ((1 - a) * img1 + a * img2)), axis=0)
        interpolated = np.concatenate((interpolated, img1), axis=0)

        if latent:
            box = interpolated.reshape((6,32)).astype(np.float32)
            recon_interp = self._session.run([self._decoder], feed_dict={self._encoder: box})[0]
            interpolated = recon_interp.reshape((6, 784))

        img_dim = np.sqrt(int(interpolated.shape[1]))
        assert int(img_dim) == img_dim

        # Display interpolated images
        canvas = np.empty((img_dim, img_dim * interp_levels))
        for j in range(interp_levels):
            canvas[:, j * img_dim:(j + 1) * img_dim] = interpolated[j].reshape([img_dim, img_dim])

        plt.figure()
        plt.title(title)
        plt.imshow(canvas, origin="upper", cmap="gray")
        plt.show()

    def pixel_interpolation(self, batch):
        # take two images from batch
        img1 = batch[0, :].reshape([1, batch.shape[1]])
        img2 = batch[1, :].reshape([1, batch.shape[1]])
        self.interpolate_images(img1, img2, "Pixel Space Interpolation")


if __name__ == '__main__':

    mnist_data = input_data.read_data_sets("MNIST", one_hot=True)
    batch, label = mnist_data.test.next_batch(256)

    with AutoEncoderNetwork(mnist_data, 784, 50, 32, batch_size=128, num_epochs=20000, learning_rate=0.05) as network:
        network.train(0.005, 185)
        cost = network.test(False)
        print("Test Error: %.4f" % cost)
        reduced_imgs = network.reduce_dim()
        network.pixel_interpolation(batch)
        network.pca_reduction(batch, label, 2, True)
