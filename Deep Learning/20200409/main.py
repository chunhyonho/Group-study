import numpy as np
import matplotlib.pyplot as plt


class BinaryInputLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        pass

    def backward(self, dout):
        pass


# Perform Add(x, y) = x + y
class AddLayer(BinaryInputLayer):
    def forward(self, x, y):
        self.x = x
        self.y = y
        output = x + y

        return output

    @staticmethod
    def backward(self, dout):
        dx = dout
        dy = dout

        return [dx, dy]


# Perform Mul(x, y) = x * y
class MultiplyLayer(BinaryInputLayer):
    def forward(self, x, y):
        self.x = x
        self.y = y
        output = x * y

        return output

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return [dx, dy]


# Perform ReLu(x) = max(0, x)
class ReLuLayer(BinaryInputLayer):
    def forward(self, x):
        self.x = x
        # use self.y as cache
        # vector/matrix of Boolean (mask)
        # ex. [[True False] [False False]]
        self.y = (x <= 0)
        # Replace neg val to 0
        output = x.copy()
        output[self.y] = 0

        return output

    def backward(self, dout):
        dx = dout
        # Replace neg val to 0
        # neg val has no impact on output
        dx[self.y] = 0

        return dx


# Perform Sigmoid(x) = 1 / 1 + exp(-x)
class SigmoidLayer(BinaryInputLayer):
    def forward(self, x):
        self.x = x
        # TODO : np.exp() is not stable because of Inf val. ( ex. exp(1000) -> RuntimeError / overflow )
        # np.exp(-x) -> np.exp(x - x.max())
        output = 1 / (1 + np.exp(-x))
        # use self.y as cache (for backprop)
        # Derivative of d Sigmoid(x) / dx = Sigmoid(x) * ( 1 - Sigmoid(x) )
        # self.y = Sigmoid(x)
        self.y = output

        return output

    def backward(self, dout):
        dx = dout * self.y * (1 - self.y)

        return dx


class SoftMaxLossLayer(BinaryInputLayer):

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z)
        y = exp_z / np.sum(exp_z)

        return y

    @staticmethod
    def cross_entropy(y, t):
        return -np.sum(t * np.log(y))

    def forward(self, x, y):
        # input x
        self.x = self.softmax(x)
        # one-hot encoding t
        self.y = y

        loss = self.cross_entropy(self.x, self.y)

        return loss

    def backward(self, dout):
        batch_size = self.y.shape[0]
        dx = (self.x - self.y) / batch_size

        return dx


class AffineLayer:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        output = np.dot(x, self.W) + self.b

        return output

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


# CODE EXAMPLE

###########################################################################
# DATA GENERATION
###########################################################################

# number of data points per class
N = 100
# dimensionality
D = 2
# number of classes
K = 3

# Data
X = np.zeros((N*K,D))
# Label
y = np.zeros(N*K, dtype='uint8')

for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

# Data visualization
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

###########################################################################

# Parameter GENERATION

h = 100
# np.random.randn => Generate Random number from N(0,1)
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

step_size = 1
reg = 1e-3

# TODO 여기서 매 루프마다 AffineLayer를 선언해서 비효율적 => update function 만들어서 넣기

num_examples = X.shape[0]
for i in range(7000):
    # Create Two Affine Layer
    hidden_l = AffineLayer(W, b)
    score_l = AffineLayer(W2, b2)

    # Execute Foward Propagation
    out_1 = hidden_l.forward(X)
    out_1 = np.maximum(0, out_1) # Perform ReLu
    out_2 = score_l.forward(out_1)

    # Provide score(out_2) to SoftMax func
    exp_l = np.exp(out_2)
    probs_l = exp_l / np.sum(exp_l, axis=1, keepdims=True)

    # Extract Probability of real class
    correct_logprobs_l = -np.log(probs_l[range(num_examples), y])
    data_loss_l = np.sum(correct_logprobs_l) / num_examples
    reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2*W2)
    loss_l = data_loss_l + reg_loss


    if i % 1000 == 0:
        print("Custom iteration %d: loss %f" % (i, loss_l))

    # Calculate Gradients
    dscores_l = probs_l
    dscores_l[range(num_examples), y] -= 1
    dscores_l /= num_examples

    dhidden_l = score_l.backward(dscores_l)
    dW2_l = score_l.dW
    db2_l = score_l.db
    dhidden_l[out_1 <= 0] = 0
    hidden_l.backward(dhidden_l)
    dW_l = hidden_l.dW
    db_l = hidden_l.db

    dW2_l += reg * W2
    dW_l += reg * W

    # Update Parameters
    W += -step_size * dW_l
    b += -step_size * db_l
    W2 += -step_size * dW2_l
    b2 += -step_size * db2_l


h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
