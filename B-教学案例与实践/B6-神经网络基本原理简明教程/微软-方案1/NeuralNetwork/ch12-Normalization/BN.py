import numpy as np

class batchNorm():
    """Batch normalization layer implemented by numpy

    This is the basic class for batch normalization layer used in deep learning

    Attributes:
        input: the name of layer before this layer
        output: the name of layer after this layer
        mean_list: the list stores the mean values of variables trained using this layer
        var_list: the list stores the variance values of variables trained using this layer
        eps: a small value to avoid the variance is zero
        gamma: a learnable gamma to do linear transformation
        beta: a learnable beta to do linear transformation
        variable: store the input variable
        gradient_gamma: the gradient to update gamma
        gradient_beta: the gradient to update beta
    """

    def __init__(self, input, output, outputChannel):
        """Construct a BN layer"""
        self.input = input
        self.output = output
        self.mean_list = []
        self.var_list = []
        self.eps = 1e-5
        self.gamma = np.ones([1, outputChannel])
        self.beta = np.zeros([1, outputChannel])

    def forward(self, variable, train=True):
        """the function used in forward process

        This function describes the process of forward calculation

        Args:
            variable: the input variable to be processed, 
                      assume the shape of this variable is [batch, channels]
            train: a bool variable represents whether it is still under training 

        Return:
            the variable after processed, [batch, channels]

        """
        if train:
            mean = np.mean(variable, axis=0)
            var = np.var(variable, axis=0)
            self.variable = variable
            variable = (variable - mean) / np.sqrt(var + self.eps)
            self.mean_list.append(mean)
            self.var_list.append(var)
        else:
            mean = sum(self.mean_list) / len(self.mean_list)
            var = sum(self.var_list) / len(self.var_list)
            var = var * self.variable.shape[0] / (self.variable.shape[0] - 1)
            variable = (variable - mean) / np.sqrt(var + self.eps)
        # end if

        return self.gamma * variable + self.beta
    
    def backward(self, error):
        """backward function used in training

        This function will compute the gradient of the gamma and beta in BN layer
        and return error to previous layer

        Args:
            error: the error returned by the later layer in the network

        Return:
            the computed error to transfer to the previous layer
        """

        batch_size = self.variable.shape[0]
        mean = self.mean_list[-1]
        var = self.var_list[-1]

        # calculate the gradient to update gamma and beta
        self.gradient_gamma = np.sum(error * ((self.variable - mean) / np.sqrt(var + self.eps)), axis=0)
        self.gradient_beta = np.sum(error, axis=0)

        # the error to normalized variable, [1, channels] .* [batch, channels] = [batch, channles]
        error_normalized_variable = self.gamma * error

        # the error related to the var of the input variable, [1, channels]
        error_var = (-0.5) * np.sum(error_normalized_variable * (self.variable - mean) , axis = 0) * ((var + self.eps) ** (-1.5))

        # the error related to the mean of the input variable, [1, channels]
        error_mean = (-1) * np.sum(error_normalized_variable / np.sqrt(var + self.eps), axis=0) + error_var * (-2) * np.sum(self.variable - mean, axis=0) / batch_size

        # sum the above error items to get the final error
        error = error_normalized_variable / np.sqrt(var + self.eps) + (error_var * 2 * (self.variable - mean) + error_mean) / batch_size

        return error

    def update(self, learning_rate=0.1):
        """update gamma and beta in BN layer

        This function will update gamma and beta in training procedure

        Args:
            learning_rate: the learning rate to update the two parameters

        Return:
            None
        """
        self.gamma -= self.gradient_gamma * learning_rate
        self.beta -= self.gradient_beta * learning_rate
# end BN class



# the following part code is copied from cs231n to check the gradient
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #######################################################################

        # Forward pass
        # Step 1 - shape of mu (D,)
        mu = 1 / float(N) * np.sum(x, axis=0)

        # Step 2 - shape of var (N,D)
        xmu = x - mu

        # Step 3 - shape of carre (N,D)
        carre = xmu**2

        # Step 4 - shape of var (D,)
        var = 1 / float(N) * np.sum(carre, axis=0)

        # Step 5 - Shape sqrtvar (D,)
        sqrtvar = np.sqrt(var + eps)

        # Step 6 - Shape invvar (D,)
        invvar = 1. / sqrtvar

        # Step 7 - Shape va2 (N,D)
        va2 = xmu * invvar

        # Step 8 - Shape va3 (N,D)
        va3 = gamma * va2

        # Step 9 - Shape out (N,D)
        out = va3 + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var

        cache = (mu, xmu, carre, var, sqrtvar, invvar,
                 va2, va3, gamma, beta, x, bn_param)
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #######################################################################
        mu = running_mean
        var = running_var
        xhat = (x - mu) / np.sqrt(var + eps)
        out = gamma * xhat + beta
        cache = (mu, var, gamma, beta, bn_param)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    ##########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    ##########################################################################
    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    # Backprop Step 9
    dva3 = dout
    dbeta = np.sum(dout, axis=0)

    # Backprop step 8
    dva2 = gamma * dva3
    dgamma = np.sum(va2 * dva3, axis=0)

    # Backprop step 7
    dxmu = invvar * dva2
    dinvvar = np.sum(xmu * dva2, axis=0)

    # Backprop step 6
    dsqrtvar = -1. / (sqrtvar**2) * dinvvar

    # Backprop step 5
    dvar = 0.5 * (var + eps)**(-0.5) * dsqrtvar

    # Backprop step 4
    dcarre = 1 / float(N) * np.ones((carre.shape)) * dvar

    # Backprop step 3
    dxmu += 2 * xmu * dcarre

    # Backprop step 2
    dx = dxmu
    dmu = - np.sum(dxmu, axis=0)

    # Basckprop step 1
    dx += 1 / float(N) * np.ones((dxmu.shape)) * dmu

    return dx, dgamma, dbeta