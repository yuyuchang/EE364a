import numpy as np
import matplotlib.pyplot as plt

def lp_acent(A, b, c, x_0):
	x_0 = x_0.reshape(len(x_0), 1)
	b = b.reshape(len(b), 1)
	c = c.reshape(len(c), 1)
	alpha = 0.01
	beta = 0.5
	epsilon = 1e-3
	max_iters = 100

	lambda_hist = np.array([])
	A = np.matrix(A)

	if min(x_0) <= 0 or np.linalg.norm(np.dot(A, x_0) - b) > epsilon:
		print("Error: x_0 is not feasible.")
		return np.array([]), np.array([]), lambda_hist

	m = b.size
	n = x_0.size
	x = x_0

	for iterNum in range(max_iters):
		H = np.diag(1 / np.power(x.reshape(n), 2))
		g = c - 1 / x

		X = np.diag(x.reshape(n) ** 2)
		w = np.linalg.lstsq(A * X * A.T, -A * X * g)[0]
		dx = -X * (A.T * w + g)

		lambdasqr = -np.dot(g.reshape(n), dx.reshape(n).T)
		lambda_hist = np.append(lambda_hist, lambdasqr / 2)
	
		if lambdasqr / 2 <= epsilon:
			return x, w, lambda_hist

		t = 1
		while min(x + t * dx) <= 0:
			t *= beta
		while t * np.dot(c.reshape(n), dx.reshape(n).T)\
			- np.sum(np.log(x.reshape(n) + t * dx.reshape(n)))\
			+ np.sum(np.log(x.reshape(n)))\
			- alpha * t * np.dot(g.T, dx) > 0:
			t *= beta
		x += t * dx
	print("Error: max_iters reached")
	return np.array([]), np.array([]), lambda_hist

if __name__ == '__main__':

	A = np.random.rand(100, 500)
	assert np.linalg.matrix_rank(A) == 100
	x_0 = np.abs(np.random.rand(500))
	b = A @ x_0
	c = np.random.rand(500)

	x_star, nu_star, lambda_hist = lp_acent(A, b, c, x_0)

	plt.figure()
	plt.plot(np.arange(len(lambda_hist)) + 1, lambda_hist)
	plt.xlabel('iterations')
	plt.ylabel('lambda square / 2')
	plt.yscale('log')
	plt.savefig('A9.5a.png')
