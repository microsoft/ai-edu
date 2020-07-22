import numpy as np
from collections import defaultdict

getW1 = [[0.236, -0.962, 0.686, 0.785, -0.454, -0.833, -0.744, 0.677, -0.427, -0.066],
		[-0.907, 0.894, 0.225, 0.673, -0.579, -0.428, 0.685, 0.973, -0.070, -0.811],
		[-0.576, 0.658, -0.582, -0.112, 0.662, 0.051, -0.401, -0.921, -0.158, 0.529],
		[0.517, 0.436, 0.092, -0.835, -0.444, -0.905, 0.879, 0.303, 0.332, -0.275],
		[0.859, -0.890, 0.651, 0.185, -0.511, -0.456, 0.377, -0.274, 0.182, -0.237],
		[0.368, -0.867, -0.301, -0.222, 0.630, 0.808, 0.088, -0.902, -0.450, -0.408],
		[0.728, 0.277, 0.439, 0.138, -0.943, -0.409, 0.687, -0.215, -0.807, 0.612],
		[0.593, -0.699, 0.020, 0.142, -0.638, -0.633, 0.344, 0.868, 0.913, 0.429],
		[0.447, -0.810, -0.061, -0.495, 0.794, -0.064, -0.817, -0.408, -0.286, 0.149]]

getW2 = [[-0.868, -0.406, -0.288, -0.016, -0.560, 0.179, 0.099, 0.438, -0.551],
		[-0.395, 0.890, 0.685, -0.329, 0.218, -0.852, -0.919, 0.665, 0.968],
		[-0.128, 0.685, -0.828, 0.709, -0.420, 0.057, -0.212, 0.728, -0.690],
		[0.881, 0.238, 0.018, 0.622, 0.936, -0.442, 0.936, 0.586, -0.020],
		[-0.478, 0.240, 0.820, -0.731, 0.260, -0.989, -0.626, 0.796, -0.599],
		[0.679, 0.721, -0.111, 0.083, -0.738, 0.227, 0.560, 0.929, 0.017],
		[-0.690, 0.907, 0.464, -0.022, -0.005, -0.004, -0.425, 0.299, 0.757],
		[-0.054, 0.397, -0.017, -0.563, -0.551, 0.465, -0.596, -0.413, -0.395],
		[-0.838, 0.053, -0.160, -0.164, -0.671, 0.140, -0.149, 0.708, 0.425],
		[0.096, -0.995, -0.313, 0.881, -0.402, -0.631, -0.660, 0.184, 0.487]]
class word2vec():

	def __init__(self):
		self.n = settings['n']
		self.lr = settings['learning_rate']
		self.epochs = settings['epochs']
		self.window = settings['window_size']

	def generate_training_data(self, settings, corpus):
		# Find unique word counts using dictonary
		word_counts = defaultdict(int)
		for row in corpus:
			for word in row:
				word_counts[word] += 1
		self.v_count = len(word_counts.keys())
		self.words_list = list(word_counts.keys())
		self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
		self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
		training_data = []
		for sentence in corpus:
			sent_len = len(sentence)
			for i, word in enumerate(sentence):
				w_target = self.word2onehot(sentence[i])
				w_context = []
				for j in range(i - self.window, i + self.window+1):
					if j != i and j <= sent_len-1 and j >= 0:
						w_context.append(self.word2onehot(sentence[j]))
				training_data.append([w_target, w_context])

		return np.array(training_data)

	def word2onehot(self, word):
		# word_vec - initialise a blank vector
		word_vec = [0 for i in range(0, self.v_count)] # Alternative - np.zeros(self.v_count)
		# Get ID of word from word_index
		word_index = self.word_index[word]

		# Change value from 0 to 1 according to ID of the word
		word_vec[word_index] = 1

		return word_vec

	def train(self, training_data):
		self.w1 = np.array(getW1)
		self.w2 = np.array(getW2)
		# self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
		# self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
		
		# Cycle through each epoch
		for i in range(self.epochs):
			# Intialise loss to 0
			self.loss = 0
			# Cycle through each training sample
			# w_t = vector for target word, w_c = vectors for context words
			for w_t, w_c in training_data:
				# Forward pass
				# 1. predicted y using softmax (y_pred) 2. matrix of hidden layer (h) 3. output layer before softmax (u)
				y_pred, h, u = self.forward_pass(w_t)

				# Calculate error
				# 1. For a target word, calculate difference between y_pred and each of the context words
				# 2. Sum up the differences using np.sum to give us the error for this particular target word
				EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
				# Backpropagation
				# We use SGD to backpropagate errors - calculate loss on the output layer 
				self.backprop(EI, h, w_t)

				# Calculate loss
				# There are 2 parts to the loss function
				# Part 1: -ve sum of all the output +
				# Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
				# Note: word.index(1) returns the index in the context word vector with value 1
				# Note: u[word.index(1)] returns the value of the output layer before softmax
				self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
			print('Epoch:', i, "Loss:", self.loss)

	def forward_pass(self, x):
		# x is one-hot vector for target word, shape - 9x1
		# Run through first matrix (w1) to get hidden layer - 10x9 dot 9x1 gives us 10x1
		h = np.dot(x, self.w1)
		# Dot product hidden layer with second matrix (w2) - 9x10 dot 10x1 gives us 9x1
		u = np.dot(h, self.w2)
		# Run 1x9 through softmax to force each element to range of [0, 1] - 1x8
		y_c = self.softmax(u)
		return y_c, h, u

	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum(axis=0)

	def backprop(self, e, h, x):
		# Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
		# Going backwards, we need to take derivative of E with respect of w2
		# h - shape 10x1, e - shape 9x1, dl_dw2 - shape 10x9
		# x - shape 9x1, w2 - 10x9, e.T - 9x1
		dl_dw2 = np.outer(h, e)
		dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

		# Update weights
		self.w1 = self.w1 - (self.lr * dl_dw1)
		self.w2 = self.w2 - (self.lr * dl_dw2)

	# Get vector from word
	def word_vec(self, word):
		w_index = self.word_index[word]
		v_w = self.w1[w_index]
		return v_w

	# Input vector, returns nearest word(s)
	def vec_sim(self, word, top_n):
		v_w1 = self.word_vec(word)
		word_sim = {}

		for i in range(self.v_count):
			# Find the similary score for each word in vocab
			v_w2 = self.w1[i]
			theta_sum = np.dot(v_w1, v_w2)
			theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
			theta = theta_sum / theta_den

			word = self.index_word[i]
			word_sim[word] = theta

		words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

		for word, sim in words_sorted[:top_n]:
			print(word, sim)

settings = {
	'window_size': 2,			# context window +- center word
	'n': 10,					# dimensions of word embeddings, also refer to size of hidden layer
	'epochs': 50,				# number of training epochs
	'learning_rate': 0.01		# learning rate
}

text = "natural language processing and machine learning is fun and exciting"

# Note the .lower() as upper and lowercase does not matter in our implementation
# [['natural', 'language', 'processing', 'and', 'machine', 'learning', 'is', 'fun', 'and', 'exciting']]
corpus = [[word.lower() for word in text.split()]]

# Initialise object
w2v = word2vec()

# Numpy ndarray with one-hot representation for [target_word, context_words]
training_data = w2v.generate_training_data(settings, corpus)

# Training
w2v.train(training_data)

# Get vector for word
word = "machine"
vec = w2v.word_vec(word)
print(word, vec)

# Find similar words
w2v.vec_sim("machine", 3)