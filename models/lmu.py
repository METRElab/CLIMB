import numpy as np
import scipy.linalg
import scipy.special
import nengo

from cereb_platform import GranuleLayer

def get_weights_for_delays(q, r):
    r = np.asarray(r)
    m = np.asarray([scipy.special.legendre(i)(2*r - 1) for i in range(q)])
    return m.reshape(q, -1).T

def create_AB(q, theta):
    A = np.zeros((q, q))
    B = np.zeros((q, 1))
    for i in range(q):
        B[i] = (-1.)**i * (2*i+1)
        for j in range(q):
            A[i,j] = (2*i+1)*(-1 if i<j else (-1.)**(i-j+1))
    return A / theta, B / theta

class LMU(GranuleLayer):
    name = 'lmu'
    defaults = dict(theta=1.8, q=10, tau_syn=0.1,
                    pfpc_learning_rate=2e-5)

    def build(self, mf):
        p = self.p
        q, tau_syn = p['q'], p['tau_syn']
        A, B = create_AB(q, p['theta'])
        n_ch = mf.dimensions
        if n_ch > 1:
            # CS, metronome: block A and B
            A = scipy.linalg.block_diag(*[A] * n_ch)
            B = scipy.linalg.block_diag(*[B] * n_ch)
        dims = q * n_ch
        lmu = nengo.Ensemble(n_neurons=p['n_granule'], dimensions=dims,
                             radius=2 * np.sqrt(n_ch), seed=1)
        if p['granule_noise_std'] > 0:
            lmu.noise = nengo.processes.WhiteNoise(
                dist=nengo.dists.Gaussian(0, p['granule_noise_std']))
        nengo.Connection(mf, lmu, transform=tau_syn * B, synapse=tau_syn)
        nengo.Connection(lmu, lmu, transform=A * tau_syn + np.identity(dims), synapse=tau_syn)
        self.granule = lmu
        return lmu
