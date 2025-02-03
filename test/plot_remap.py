import matplotlib.pyplot as plt
import numpy as np
from gllf.gllf_layer import gllf_layer_radial
from gllf.gllf_utils import *

max_intensities = 4
threshold = 0.5

i = np.linspace(-3,3,100,dtype=np.float32)
g = np.arange(max_intensities) / max_intensities

sigmoid = lambda x:1/(1+np.exp(-x))
soft_switch = lambda x, x0, x1, k: sigmoid(k*(x-x0)) * (1 - sigmoid(k*(x-x1)))

def remap_llf(i, g, t, alpha, beta):
    diff = i - g
    detail = g + np.sign(diff) * t * np.pow(np.abs(diff) / t, alpha)
    # detail = g + alpha * (diff-t) * np.exp(-(diff-t)**2/2.)
    compress =  g + np.sign(diff) * (beta * (np.abs(diff) - t) + t)
    # return detail
    return np.where(np.abs(diff) < t, detail, compress)


def remap_ours(i, g, t, alpha, beta):
    diff = i - g
    # detail = g + np.sign(diff) * t * np.pow(np.abs(diff) / t, alpha)
    return g + beta * diff + alpha * diff * np.exp(-diff**2/(t*2.))
    detail = g + beta * diff + alpha * diff * np.exp(-(diff)**2/2.)
    # compress =  g + np.sign(diff) * t*(1 -beta) + beta * diff
    
    # return (soft_switch(10))*detail + (1-soft_switch(10)) * compress
    return detail
    return np.where(np.abs(diff) < t, detail, compress)

def odd_monotonic_fourier(i, g, terms=10, alpha=2):
    """
    Differentiable odd, monotonically increasing function using Fourier basis.
    
    Parameters:
        x (array): Input values.
        terms (int): Number of Fourier terms to include.
        alpha (float): Decay rate for coefficients (must be > 1 for convergence).
    
    Returns:
        array: Evaluated Fourier series.
    """
    x = i - g
    f_x = np.zeros_like(x)
    for n in range(1, terms + 1):
        b_n = 1 / n**alpha  # Exponentially decaying coefficients
        f_x += (b_n / n) * np.sin(n * x)  # Odd function using sine terms
    return f_x


def gaussian_rbf(x, c, sigma):
    """Gaussian radial basis function."""
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))

def odd_gaussian_rbf(x, c, sigma):
    """Odd Gaussian radial basis function."""
    return gaussian_rbf(x, c, sigma) - gaussian_rbf(x, -c, sigma)

def monotonic_odd_function(x, centers, sigma, weights):
    """
    Monotonically increasing odd function using Gaussian RBFs.
    
    Parameters:
        x (array): Input values.
        centers (array): Centers of the Gaussian RBFs.
        sigma (float): Width of the Gaussian RBFs.
        weights (array): Weights for the RBFs.
    
    Returns:
        array: Evaluated function.
    """
    f_x = np.zeros_like(x)
    for c, w in zip(centers, weights):
        # f_x += w * odd_gaussian_rbf(x, c, sigma)
        f_x += w * gaussian_rbf(x, c, sigma)
    return f_x

def rbf_plot(i, debug=False):
    max_levels=4.
    level = 2. / max_levels
    basis_ct = 4
    sigma = np.array([0.] * basis_ct)
    w = np.array([0.] * basis_ct)
    sigma[0] = 1
    w[0] = 1
    if(debug):
        return level + radial_basis(i[None,:], level, w[:,None], sigma[:,None], 'gaussian_1d')
    else:
        return level + radial_basis(i[None,:], level, w[:,None], sigma[:,None], 'gaussian_1d')
    
# r_exp = rbf.gaussian_radial_basis(i, [0.5], t, w)
# switch = soft_switch(i, 0.5, 1, 100)

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

# plt.plot(i,r,'b')
# plt.plot(i,r_ours,'r')
# plt.plot(i,r_sine,'r')
# plt.plot(i,rbf_plot(i, True)[0],'r')
# plt.plot(i,rbf_plot(i, True)[1],'g')
ax1.plot(i,rbf_plot(i)[0],'b')
ax2.plot(i,rbf_plot(i)[0],'b')
ax2.set_xlim([0,1])
ax2.set_ylim([0,1])
# plt.plot(i,switch,'g')
# plt.legend()
plt.savefig('./remap.png')