"""
Lhs functions are inspired by
https://github.com/clicumu/pyDOE2/blob/
master/pyDOE2/doe_lhs.py
"""
import numpy as np
from scipy import spatial
from tqdm.auto import tqdm
from skopt.space import Space, Categorical
from skopt.sampler.base import InitialPointGenerator
import cupy as cp

xp = cp

def check_random_state(seed):
    if isinstance(seed, cp.random.RandomState):
        rng = seed
    elif np.isscalar(seed):
        rng = cp.random.RandomState(seed=int(seed))
    else:
        rng = cp.random.RandomState(seed=None)
    return rng

def _random_permute_matrix(h, random_state=None):
    xp = cp.get_array_module(h)
    rng = check_random_state(random_state)
    h_rand_perm = xp.zeros_like(h)
    samples, n = h.shape
    for j in range(int(n)):
        order = rng.permutation(int(samples))
        h_rand_perm[:, j] = h[order, j]
    return h_rand_perm


class Lhs(InitialPointGenerator):
    """Latin hypercube sampling

    Parameters
    ----------
    lhs_type : str, default='classic'
        - 'classic' - a small random number is added
        - 'centered' - points are set uniformly in each interval

    criterion : str or None, default='maximin'
        When set to None, the LHS is not optimized

        - 'correlation' : optimized LHS by minimizing the correlation
        - 'maximin' : optimized LHS by maximizing the minimal pdist
        - 'ratio' : optimized LHS by minimizing the ratio
          `max(pdist) / min(pdist)`

    iterations : int
        Defines the number of iterations for optimizing LHS
    """

    def __init__(
        self, lhs_type="classic", criterion=None, iterations=1000, verbose=True
    ):
        self.lhs_type = lhs_type
        self.criterion = criterion
        self.iterations = iterations
        self.verbose = verbose

    def generate(self, dimensions, n_samples, batch_size=None, random_state=None, distance_metric='euclidean'):
        """Creates latin hypercube samples.

        Parameters
        ----------
        dimensions : list, shape (n_dims,)
            List of search space dimensions.
            Each search dimension can be defined either as

            - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
              dimensions),
            - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
              dimensions),
            - as a list of categories (for `Categorical` dimensions), or
            - an instance of a `Dimension` object (`Real`, `Integer` or
              `Categorical`).

        n_samples : int
            The order of the LHS sequence. Defines the number of samples.
        random_state : int, RandomState instance, or None (default)
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        np.array, shape=(n_dim, n_samples)
            LHS set
        """
        if batch_size is None:
            batch_size = n_samples
        batch_start_indices = np.arange(0, n_samples, batch_size)
        batch_idx = 0

        rng = check_random_state(random_state)
        space = Space(dimensions)
        transformer = space.get_transformer()
        n_dim = space.n_dims
        space.set_transformer("normalize")
        if self.criterion is None or n_samples == 1:
            h = self._lhs_normalized(n_dim, n_samples, rng).get()
            h = space.inverse_transform(h)
            space.set_transformer(transformer)
            return h
        else:
            iters = range(self.iterations)
            if self.verbose:
                iters = tqdm(
                    iters,
                    total=self.iterations,
                    desc=f"LHS|{self.criterion}|{self.lhs_type}|{batch_idx}",
                    dynamic_ncols=True
                )
            h_opt = [None]*len(batch_start_indices)
            for i in iters:
                mincorr = np.full((len(batch_start_indices),), np.inf)
                maxdist = np.full((len(batch_start_indices),), 0)
                minratio = np.full((len(batch_start_indices),), np.inf)
                for batch_idx, start_idx in enumerate(batch_start_indices):
                    if self.verbose:
                        iters.set_description(
                            f"LHS|{self.criterion}|{self.lhs_type}|{batch_idx}",
                            refresh=True
                        )
                    stop_idx = min(start_idx + batch_size, n_samples)
                    n_samples_batch = stop_idx - start_idx
                    if self.criterion == "correlation":
                        # Generate a random LHS
                        h_batch = cp.asarray(self._lhs_normalized(n_dim, n_samples_batch, rng))
                        r = cp.corrcoef(h_batch).get()
                        if (
                            len(np.abs(r[r != 1])) > 0
                            and np.max(np.abs(r[r != 1])) < mincorr[i]
                        ):
                            mincorr[i] = np.max(np.abs(r - np.eye(r.shape[0])))
                            h_opt[i] = h_batch.get()
                    elif self.criterion == "maximin":
                        # Maximize the minimum distance between points
                        h_batch = self._lhs_normalized(n_dim, n_samples_batch, rng)
                        if isinstance(h_batch, cp.ndarray):
                            h_batch = h_batch.get()
                        d = spatial.distance.pdist(np.array(h_batch), distance_metric)
                        if maxdist[i] < np.min(d):
                            maxdist[i] = np.min(d)
                            h_opt[i] = h_batch
                    elif self.criterion == "ratio":
                        # Maximize the minimum distance between points
                        h_batch = self._lhs_normalized(n_dim, n_samples_batch, rng)
                        if isinstance(h_batch, cp.ndarray):
                            h_batch = h_batch.get()
                        p = spatial.distance.pdist(np.array(h_batch), "euclidean")
                        if np.min(p) == 0:
                            ratio = np.max(p) / 1e-8
                        else:
                            ratio = np.max(p) / np.min(p)
                        if minratio[i] > ratio:
                            minratio[i] = ratio
                            h_opt[i] = h_batch
                    else:
                        raise ValueError("Wrong criterion." "Got {}".format(self.criterion))
                h_opt = np.vstack(h_opt)
            h_opt = space.inverse_transform(h_opt)
            space.set_transformer(transformer)
            return h_opt

    def _lhs_normalized(self, n_dim, n_samples, random_state):
        rng = check_random_state(random_state)
        x = cp.linspace(0, 1, n_samples + 1)
        u = rng.rand(n_samples, n_dim)
        h = cp.zeros_like(u)
        if self.lhs_type == "centered":
            for j in range(n_dim):
                h[:, j] = cp.diff(x) / 2.0 + x[:n_samples]
        elif self.lhs_type == "classic":
            for j in range(n_dim):
                h[:, j] = u[:, j] * cp.diff(x) + x[:n_samples]
        else:
            raise ValueError("Wrong lhs_type. Got ".format(self.lhs_type))
        return _random_permute_matrix(h, random_state=rng)
