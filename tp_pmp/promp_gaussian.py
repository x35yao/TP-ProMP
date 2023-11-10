import numpy as np
import scipy
from .utils import make_block_diag, force_sym

def vm(t, params, conf):
    """ A set of Von-Mises basis functions in one dimension
    """
    sigma_sq = params[0]
    centers = params[1:]
    ans = np.exp(np.cos(2 * np.pi * (t - centers)) / sigma_sq)
    return ans

def sqexp(t, params, conf):
    """ A set of radial basis functions in one dimension
    """
    sigma_sq = np.exp(params[0])**2
    centers = params[1:]
    ans = np.exp(-0.5*(t - centers)**2 / sigma_sq)
    return ans

def poly(t, params, conf):
    """ Polynomial with order equal to dim-1
    """
    order = conf['order']
    basis_f = [t**ix for ix in range(order+1)]
    return np.array(basis_f)

def comb_basis(t, params, conf):
    '''
    Compute basis function value at time t given parameters
    '''
    basis = {"sqexp": sqexp, "poly": poly, "vm":vm}
    ans = []
    start = 0
    for c in conf:
        end = start + c['nparams']
        ans.append(basis[c['type']](t, params[start:start+end], conf=c['conf']))
        start = end
    return np.concatenate(ans)

def cov_mat_precomp(cov_mat):
    tmp, log_det = np.linalg.slogdet(cov_mat)
    result = {'inv': np.linalg.inv(cov_mat),
        'log_det': log_det}
    return result

def quad(a,X):
    """ Computes a quadratic form as a^T X a
    """
    return np.dot(a, np.dot(X, a))

class FullProMP():
    def __init__(self, basis, n_dims = 7, n_rfs = 1, tol = 1e-1, reg_covar = 1e-3, random_state = 1, q = True, qd = False, init_mu_w = None, init_Sigma_w = None, init_Sigma_y = None ):
        self.basis = basis
        self.n_dims = n_dims
        self.n_rfs = n_rfs
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.basis_conf = basis['conf']
        self.basis_params = np.array(basis['params'])
        self.basis_fun = lambda t, params: comb_basis(t, params, conf=self.basis_conf)
        self.q = q
        self.qd = qd
        self.bfun = self._get_bfun_lambdas( self.basis_fun, self.basis_params, q = self.q, qd = self.qd)
        self.y_dim, self.w_dim = np.shape(self.__comp_Phi_t(0.0, 10., self.bfun))
        self.init_mu_w = init_mu_w
        self.init_Sigma_w = init_Sigma_w
        self.init_Sigma_y = init_Sigma_y

    def _get_bfun_lambdas(self, basis_fun, basis_params, q=True, qd=False):
        '''
        Get basis functions for original data or second order derivative of the original data

        Parameters
            basis_fun : function objects
                Basis functions for original data
            basis_params : array-like
                Function parameters
            q : bool
                whether or not to compute the basis functions for original data
            qd: bool
                whether or not to compute the basis functions for original data
        '''
        f = lambda z: basis_fun(z, basis_params)
        bfun = {}
        if q:
            bfun['fpos'] = f
        if qd:
            bfun['fvel'] = autograd.jacobian(f)
        return bfun


    def __comp_Phi_t(self, t, T, bfun):
        """
        Computer Phi value given current time t and basis functions
        Parameters
            t : float
                Current time
            T : float
                End time
            bfun: dict
                A dictionary that contains basis functions
        """
        vals = {}
        if 'fpos' in bfun: vals['pos'] = bfun['fpos'](t/T)
        if 'fvel' in bfun: vals['vel'] = bfun['fvel'](t/T)
        if 'facc' in bfun: vals['acc'] = bfun['facc'](t/T)
        return self.__get_Phi_t(t,T,vals)

    def __get_Phi_t(self, t, T, vals):
        """
        Stack the Phi matrix for different dimension to get a block diagonal matrix.

        Parameters
            t : float
                Current time
            T : float
                End time
            vals: dict
                Phi matrix at time t for 'fpos', 'fvel', and 'facc'
        """
        assert t>=0 and t<=T
        vel_fac = 1.0/T
        pos_t = []
        vel_t = []
        acc_t = []
        for d in range(self.n_dims):
            if 'pos' in vals: pos_t.append( vals['pos'] )
            if 'vel' in vals: vel_t.append( vel_fac * vals['vel'] )
            if 'acc' in vals: acc_t.append( vel_fac**2 * vals['acc'] )
        ans = []
        if 'pos' in vals: ans.append(scipy.linalg.block_diag(*pos_t))
        if 'vel' in vals: ans.append(scipy.linalg.block_diag(*vel_t))
        if 'acc' in vals: ans.append(scipy.linalg.block_diag(*acc_t))
        return np.concatenate(ans, axis=0)

    def get_Phi(self, times):
        """ Builds a list with all the matrices Phi_t already pre-computed
        """
        Phi = []
        for time in times:
            Tn = len(time)
            duration = time[-1] - time[0]
            Phi_n = []
            for t in range(Tn):
                curr_time = time[t] - time[0]
                phi_nt = self.__comp_Phi_t(curr_time, duration, self.bfun)
                Phi_n.append(phi_nt)
            Phi.append(Phi_n)
        return Phi

    def _get_y_t(self, params):
        """ Builds the vector y_t to be compatible with the matrix Phi_t

        This method builds a vector y_t with any valid combination of
        joint position, velocity and acceleration.
        """
        y_t = []
        if 'q' in params: y_t.extend(params['q'])
        if 'qd' in params: y_t.extend(params['qd'])
        if 'qdd' in params: y_t.extend(params['qdd'])
        return np.array(y_t)


    def get_Y(self, times, q, qd):
        Y = []
        N = len(times)
        for n in range(N):
            y_n = []
            for t in range(len(times[n])):
                inst = {}
                if q is not None:
                    inst['q'] = q[n][t, :]
                if qd is not None:
                    inst['qd'] = qd[n][t, :]
                y_n.append(self._get_y_t(inst))
            Y.append(np.array(y_n))
        return Y

    def __em_lb_likelihood(self, expectations):
        #2) Load values in some variables
        w_means = expectations['w_means']
        w_covs = expectations['w_covs']
        log_det_sig_w = self.Sigma_w_val['log_det']
        inv_sig_w = self.Sigma_w_val['inv']
        log_det_sig_y = self.Sigma_y_val['log_det']
        inv_sig_y = self.Sigma_y_val['inv']
        #3) Actually compute lower bound
        ans = 0.0
        for n in range(len(self.times)):
            Tn = len(self.times[n])
            lpw = log_det_sig_w + np.trace(np.dot(inv_sig_w,w_covs[n])) + quad(w_means[n]- self.mu_w, inv_sig_w)
            lhood = 0.0
            for t in range(Tn):
                phi_nt = self.Phi[n][t]
                y_nt = self.Y[n][t]
                lhood = lhood + log_det_sig_y + quad(y_nt-np.dot(phi_nt,w_means[n]),inv_sig_y) + \
                        np.trace(np.dot(inv_sig_y, np.dot(phi_nt, np.dot(w_covs[n], phi_nt.T))))
            ans = ans + lpw + lhood
        return -0.5*ans

    def __EM_lowerbound(self, expectations, with_prior = True):
        """ Computes the EM lowerbound
        Receives a list of time vectors from the training set, the expectations computed in the
        E-step of the algorithm, and a list of optional arguments. As an optional argument eigther
        the angle positions, velocities or accelerations of the training set should be included.
        The optional arguments can also specify any of the parameters that are being optimized as
        a special value.
        """
        # Load useful variables (Including likelihood)
        inv_sig_w = self.Sigma_w_val['inv']
        log_det_sig_w = self.Sigma_w_val['log_det']
        lhood_lb = self.__em_lb_likelihood(expectations)
        if not with_prior:
            return lhood_lb
        else:
            # Compute prior log likely-hood
            lprior = 0.0
            if self.prior_mu_w is not None:
                m0 = self.prior_mu_w['m0']
                inv_V0 = self.prior_mu_w['k0']*inv_sig_w #Normal-Inverse-Wishart prior
                lprior = lprior + quad(self.mu_w-m0, inv_V0)
            if self.prior_Sigma_w is not None:
                v0 = self.prior_Sigma_w['v']
                D = np.shape(self.Sigma_w)[0]
                if 'mean_cov_mle' in self.prior_Sigma_w:
                    S0 = self.prior_Sigma_w['mean_cov_mle'](self.__Sigma_w_mle) * (v0 + D + 1)
                else:
                    S0 = self.prior_Sigma_w['invS0']
                lprior = lprior + (v0 + D + 1)*log_det_sig_w + np.trace(np.dot(S0, inv_sig_w))
            # Compute full lower bound
            return -0.5*lprior + lhood_lb

    def initialize(self):
        self.mu_w = np.zeros(self.w_dim)
        self.Sigma_w = np.eye(self.w_dim, self.w_dim)
        self.Sigma_y = np.eye(self.y_dim, self.y_dim)

    def _e_step(self, Y = None, with_prior = False):
        if Y is None:
            Y = self.Y
        # 1) Computer matrix inverse
        self.Sigma_w_val = cov_mat_precomp(self.Sigma_w)
        self.Sigma_y_val = cov_mat_precomp(self.Sigma_y)
        inv_sig_w = self.Sigma_w_val['inv']
        inv_sig_y = self.Sigma_y_val['inv']
        # 2) Compute expectations
        w_means = []
        w_covs = []
        for n, time in enumerate(self.times):
            Tn = len(Y[n])
            sum_mean = np.dot(inv_sig_w, self.mu_w)
            sum_cov = inv_sig_w
            for t in range(Tn):
                phi_nt = self.Phi[n][t]
                tmp1 = np.dot(np.transpose(phi_nt), inv_sig_y)
                sum_mean = sum_mean + np.dot(tmp1, self.Y[n][t])
                sum_cov = sum_cov + np.dot(tmp1, phi_nt)
            Swn = force_sym(np.linalg.inv(sum_cov))
            wn = np.dot(Swn, sum_mean)
            w_means.append(wn)
            w_covs.append(Swn)
        expectations = {'w_means': w_means, 'w_covs': w_covs}
        lowerbound = self.__EM_lowerbound(expectations, with_prior)
        return lowerbound, {'w_means': w_means, 'w_covs': w_covs}

    def _m_step(self, expectations):
        w_covs = expectations['w_covs']
        w_means = expectations['w_means']
        N = len(self.times)
        mu_w, Sigma_w, Sigma_y, alpha = [], [], [], []
        n_var = lambda X: sum([np.outer(x, x) for x in X])

        # 1) Optimize mu_w
        wn_sum = sum(w_means)
        if self.prior_mu_w is not None:
            mu_w = (wn_sum + self.prior_mu_w['k0'] * self.prior_mu_w['m0']) / (N + self.prior_mu_w['k0'])
        else:
            mu_w = (wn_sum) / N

        # 2) Optimize Sigma_w
        diff_w = [x - mu_w for x in w_means]
        if self.no_Sw:
            sw_sum = force_sym(n_var(diff_w))
        else:
            sw_sum = force_sym(sum(w_covs) + n_var(diff_w))
        sw_sum = make_block_diag(sw_sum, self.n_rfs)
        self.__Sigma_w_mle = sw_sum / N  # Maximum likelyhood estimate for Sigma_w
        if self.prior_Sigma_w is not None:
            v0 = self.prior_Sigma_w['v']
            D = np.shape(self.Sigma_w)[0]
            if 'mean_cov_mle' in self.prior_Sigma_w:
                S0 = self.prior_Sigma_w['mean_cov_mle'](self.__Sigma_w_mle) * (v0 + D + 1)
            else:
                S0 = self.prior_Sigma_w['invS0']
            Sigma_w = (S0 + sw_sum) / (N + v0 + D + 1)
        else:
            Sigma_w = self.__Sigma_w_mle

        # 3) Optimize Sigma_y
        diff_y = []
        uncert_w_y = []
        for n in range(N):
            for t in range(len(self.times[n])):
                diff_y.append(self.Y[n][t] - np.dot(self.Phi[n][t], w_means[n]))
                uncert_w_y.append(np.dot(np.dot(self.Phi[n][t], w_covs[n]), self.Phi[n][t].T))
        if self.no_Sw:
            Sigma_y = (n_var(diff_y)) / len(diff_y)
        else:
            Sigma_y = (n_var(diff_y) + sum(uncert_w_y)) / len(diff_y)

        # 4) Update
        self.mu_w = mu_w
        self.Sigma_w = force_sym(Sigma_w)
        self.Sigma_w = make_block_diag(self.Sigma_w, self.n_rfs)
        if self.diag_sy:
            self.Sigma_y = np.diag(np.diag(Sigma_y))
        else:
            self.Sigma_y = force_sym(Sigma_y)

    def _get_parameters(self):
        return (
            self.mu_w,
            self.Sigma_w,
            self.Sigma_y
        )

    def _set_parameters(self, params):
        (
            self.mu_w,
            self.Sigma_w,
            self.Sigma_y
        ) = params

    def train(self, times, data, qd = None, max_iter = 30, diag_sy = True, no_Sw = False, prior_Sigma_w = None, prior_mu_w = None, print_lowerbound = False, n_init = 1):
        self.diag_sy = diag_sy
        self.no_Sw = no_Sw
        self.prior_Sigma_w = prior_Sigma_w
        self.prior_mu_w = prior_mu_w
        self.converged = False
        max_lower_bound = -np.inf
        self.Y = self.get_Y(times, data, qd)
        self.Phi = self.get_Phi(times)
        self.times = times
        for init in range(n_init):
            if (self.init_mu_w is None) or (self.init_Sigma_w is None) or (self.init_Sigma_y is None):
                self.initialize()
            else:
                self.mu_w = self.init_mu_w
                self.Sigma_w = self.init_Sigma_w
                self.Sigma_y = self.init_Sigma_y
            self.__Sigma_w_mle = self.Sigma_w
            lower_bound = -np.inf
            for it in range(max_iter):
                prev_lower_bound = lower_bound
                lower_bound, expectations = self._e_step()
                self._m_step(expectations)
                if print_lowerbound:
                    print(lower_bound)
                change = lower_bound - prev_lower_bound
                if abs(change) < self.tol:
                    self.converged = True
                    break
            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = it
        if not self.converged and max_iter > 0:
            print(
                "Initialization %d did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data." % (init + 1)
            )
        self._set_parameters(best_params)
        self.n_iter = best_n_iter
        self.lower_bound = max_lower_bound


    def marginal_w(self, time):
        phi_n = self.get_Phi([time])[0]
        means, covs= [], []
        for phi_t in phi_n:
            means.append(np.dot(phi_t, self.mu_w))
            covs.append(np.dot(np.dot(phi_t,self.Sigma_w),phi_t.T))
        return np.array(means), np.array(covs)

    def condition(self, t, T, q, Sigma_q=None, ignore_Sy=True):
        """ Conditions the ProMP

        Condition the ProMP to pass be at time t with some desired position and velocity. If there is
        uncertainty on the conditioned point pass it as the optional matrices Sigma_q,
        Sigma_qd.
        """
        times = [[0, t, T]]
        _Phi = self.get_Phi(times)
        phi_t = _Phi[0][1]
        d, lw = phi_t.shape
        inst = {'q': q}
        mu_q = self._get_y_t(inst)
        if ignore_Sy:
            tmp1 = np.dot(self.Sigma_w, phi_t.T)
            tmp2 = np.dot(phi_t, np.dot(self.Sigma_w, phi_t.T))
            tmp2 = np.linalg.inv(tmp2)
            tmp3 = np.dot(tmp1, tmp2)
            mu_w = self.mu_w + np.dot(tmp3, (mu_q - np.dot(phi_t, self.mu_w)))
            tmp4 = np.eye(d)
            if Sigma_q is not None:
                tmp4 -= np.dot(Sigma_q, tmp2)
            Sigma_w = self.Sigma_w - np.dot(tmp3, np.dot(tmp4, tmp1.T))
        else:
            inv_Sig_w = np.linalg.inv(self.Sigma_w)
            inv_Sig_y = np.linalg.inv(self.Sigma_y)
            Sw = np.linalg.inv(inv_Sig_w + np.dot(phi_t.T, np.dot(inv_Sig_y, phi_t)))
            A = np.dot(np.dot(Sw, phi_t.T), inv_Sig_y)
            b = np.dot(Sw, np.dot(inv_Sig_w, self.mu_w))
            mu_w = np.dot(A, mu_q) + b
            if Sigma_q is not None:
                Sigma_w = Sw + np.dot(A, np.dot(Sigma_q, A.T))
            else:
                Sigma_w = Sw

        self.mu_w = mu_w
        self.Sigma_w = Sigma_w

    def _n_parameters(self):
        '''Return the number of free parameters in the model'''
        mean_params = self.w_dim
        cov_params = self.w_dim * (self.w_dim + 1) / 2
        return int(cov_params + mean_params)

