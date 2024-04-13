#gradient_descent
import numpy as np
from matplotlib import pyplot as plt

# this is the cost-function
def f(x):
	return x * x
	
def df(x):
	return 2 * x

def gradient_descent(start, end, n, alpha):
	x_values = []
	y_values = []
	x = np.random.uniform(start, end)
	
	for i in range(n):
		x = x - alpha * df(x)
		x_values.append(x)
		y_values.append(f(x))
		print('#%d f(%s) = %s' % (i, x, f(x)))
	return [x_values, y_values]

if __name__ == '__main__' :
	solutions, scores = gradient_descent(-1, 1, 50, 0.1)
	inputs = np.arange(-1, 1.1, 0.1)
	plt.plot(inputs, f(inputs))
	plt.show()
	
#gradient_descent with momentum

import numpy as np
from matplotlib import pyplot as plt

# this is the cost-function
def f(x):
	return x * x
	
def df(x):
	return 2 * x

def gradient_descent(start, end, n, alpha, momentum):
	x_values = []
	y_values = []
	x = np.random.uniform(start, end)
	
	for i in range(n):
		x = x - alpha * df(x) - momentum * x
		x_values.append(x)
		y_values.append(f(x))
		print('#%d f(%s) = %s' % (i, x, f(x)))
	return [x_value, y_values]

if __name__ == '__main__' :
	solutions, scores = gradient_descent(-1, 1, 50, 0.1, 0.2)
	inputs = np.arange(-1, 1.1, 0.1)
	plt.plot(inputs, f(inputs))
	plt.show()
	
# stochastic gradient_descent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

# if batch_size = number of samples (data points) = batch gradient descent 
# if batch_size = 1 = stochastic gradient descent 
# if batch_size = [2, n]= mini-batch gradient descent
def sgd(x_values, y_values, alpha=0.01, epoch=20, batch_size=3):
	# initial parameters for slope and intercept
	m, b = 0.5, 0.5
	# we want to store the mean squared error terms (MSE)
	error = []
	
	for _ in range(epoch):
		indexes = np.random.randint(0, len(x_values), batch_size)
		xs = np.take(x_value, indexes)
		ys = np.take(y_value, indexes)
		n = len(xs)
		#print(indexes)
		f = (b + m * xs) - ys
		
		m+= -alpha * 2 * xs.dot(f).sum() / n
		b+= -alpha * 2 * f.sum() / n
		
		error.append(mean_squared_error (y, b + m *x))

	return m, b, error

def plot_regression(x_values, y_values, y_predictions):
	plt.figure(figsize=(8, 6))
	plt.title('Regression with Stochastic Gradient Descent (SGD)')
	plt.scatter(x_values, y_values, label='Data Points')
	plt.plot(x_values, y_predictions, c='#FFA35B', label='Regession')
	plt.legend(fontsize=10)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

def plot_mse(mse_values):
	plt.figure(figsize=(8, 6))
	plt.plot(range(len(mse_values)), mse_values)
	plt.title('Stochastic Gradient Descent Error')
	plt.xlabel('Epochs')
	plt.ylabel('MSE')
	plt.show()

if __name__ == '__main__':
	x = pd.Series([1, 2, 3, 5, 6, 7, 8])
	y = pd.Series([1, 1.5, 3.5, 5.2, 7.9, 6])
	slope, intercept, mses =sgd(x, y, alpha=0.01. epoch=1000, batch_size=3)
	model_predictions = intercept + slope * x

	print('Slope and intercept: %s - %s' % (slope, intercept))
	print('MSE: %s' # mean_squared_error(y, model_predictions))
	plot_regression(x, y, model_predictions)
	plot_mse(mses)

import numpy as np

# cost-function
def f(x, y):
	return x * x + y * y + 5

# derivative of the cost-function
def df(x,y):
	return np.asarray([2.0 * x, 2.0 * y])

def adam(bounds, n, alpha, beta1, beta2, epsilon=1e-8):
	x = np.asarray([0.8, 0.9])
	# initialize first moment and second moment
	m = [0.0 for _ in range(bounds.shape[0])]
	v = [0.0 for _ in range(bounds.shape[0])]
	
	for t in range(1, n+1):
		g = df(x[0], x[1])
		for i in range(x.shape[0]):
			m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
			v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] ** 2
			m_corrected = m[i] / (1.0 - beta1 ** t)
			v_corrected = v[i] / (1.0 - beta2 ** t)
			x[i] = x[i] - alpha * m_corrected / (np.sqrt(v_corrected - epsilon))
		print('(%s) - function value: %s' % (x, f(x[0], x[1])))

if __name__ == '__main__':\
	adam(np.asarray([[-1.0, 1.0], [-1.0, 1.0]]), 100, 0.05, 0.9, 0.999)
