import numpy as np
import scipy
from .utils import make_block_diag, force_sym
from scipy.special import logsumexp
from tp_pmp import promp_gaussian as promp_gaussian
from scipy.stats import multivariate_normal

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

def cov_mat_precomp(cov_mat_list):
    result = []
    for cov_mat in cov_mat_list:
        tmp, log_det = np.linalg.slogdet(cov_mat)
        result.append({'inv': np.linalg.inv(cov_mat),
            'log_det': log_det})
    return result

def quad(a,X):
    """ Computes a quadratic form as a^T X a
    """
    return np.dot(a, np.dot(X, a))

class FullProMP():
    def __init__(self,basis, n_dims = 7, n_components = 1, n_rfs = 4, tol = 1e-3, reg_covar = 1e-6, covariance_type = 'diag', random_state = 1, q = True, qd = False ):
        self.basis = basis
        self.n_dims = n_dims
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.basis_conf = basis['conf']
        self.basis_params = np.array(basis['params'])
        self.basis_fun = lambda t, params: comb_basis(t, params, conf=self.basis_conf)
        self.q = q
        self.qd = qd
        self.bfun = self._get_bfun_lambdas( self.basis_fun, self.basis_params, q = self.q, qd = self.qd)
        self.y_dim, self.w_dim = np.shape(self.__comp_Phi_t(0.0, 10., self.bfun))
        self.n_rfs = n_rfs
        self.block_dim = int(self.w_dim / self.n_rfs)

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

    def initialize(self):
        self.mu_w = np.tile(np.zeros(self.w_dim), (self.n_components, 1))
        self.Sigma_w = np.tile(np.eye(self.w_dim, self.w_dim), (self.n_components, 1, 1))
        self.Sigma_y = np.tile(np.eye(self.y_dim, self.y_dim), (self.n_components, 1, 1))
        alpha = np.random.sample(size = self.n_components)
        self.alpha = alpha / sum(alpha)

    def _estimate_log_gaussian_likelihood(self, expectations):
        # 1) Load values in some variables
        w_means = expectations['w_means']
        w_covs = expectations['w_covs']
        # 2) Actually compute lower bound
        result = []
        for n in range(len(self.times)):
            demo_likelihood = []
            Tn = len(self.times[n])
            for i in range(self.n_components):
                lpw = self.Sigma_w_val[i]['log_det'] + quad(w_means[n] - self.mu_w[i],
                                                            self.Sigma_w_val[i]['inv']) + np.trace(
                    np.dot(self.Sigma_w_val[i]['inv'], w_covs[n]))
                lhood = 0.0
                for t in range(Tn):
                    phi_nt = self.Phi[n][t]
                    y_nt = self.Y[n][t]
                    lhood = lhood + self.Sigma_y_val[i]['log_det'] + quad(y_nt - np.dot(phi_nt, w_means[n]),
                                                                          self.Sigma_y_val[i]['inv']) + \
                            np.trace(
                                np.dot(self.Sigma_y_val[i]['inv'], np.dot(phi_nt, np.dot(w_covs[n], phi_nt.T))))
                demo_likelihood.append(-0.5 * (lpw + lhood))
            result.append(demo_likelihood)
        return np.array(result)

    def _estimate_log_gaussian_lb(self, expectations, with_prior=False):
        # 1) Load useful variables (Including likelihood)
        inv_sig_w = np.array([temp['inv'] for temp in self.Sigma_w_val])
        log_det_sig_w = np.array([temp['log_det'] for temp in self.Sigma_w_val])
        # 2) Compute prior log likely-hood
        lhood_lb = self._estimate_log_gaussian_likelihood(expectations)
        # print(lhood_lb, 'likelihood')
        if not with_prior:
            # likelihood without prior
            return lhood_lb
        else:
            # likelihood with prior
            lpriors = []
            for i in range(self.n_components):
                lprior = 0.0
                if self.prior_mu_w is not None:
                    m0 = self.prior_mu_w['m0']
                    inv_V0 = self.prior_mu_w['k0'] * inv_sig_w[i]  # Normal-Inverse-Wishart prior
                    lprior = lprior + quad(self.mu_w[i] - m0, inv_V0)
                if self.prior_Sigma_w is not None:
                    prior_Sigma_w = self.prior_Sigma_w
                    v0 = prior_Sigma_w['v']
                    D = int(np.shape(self.Sigma_w[i])[0] / self.n_rfs)
                    if 'mean_cov_mle' in prior_Sigma_w:
                        S0 = prior_Sigma_w['mean_cov_mle'](self.__Sigma_w_mle[i]) * (v0 + D + 1)
                    else:
                        S0 = prior_Sigma_w['invS0']
                    lprior = lprior + (v0 + D + 1) * log_det_sig_w[i] + np.trace(np.dot(S0, inv_sig_w[i]))
                    lpriors.append(-0.5 * lprior)
            # print(lpriors, 'prior')
            full_lhood = lhood_lb + np.array(lpriors)
            # print(full_lhood, 'full_lhood')

            return full_lhood

    def _estimate_weighted_log_prob(self, expectations, with_prior = False):
        weighted_log_prob = self._estimate_log_gaussian_lb(expectations, with_prior = with_prior) + np.log(self.alpha)
        return weighted_log_prob

    def _estimate_log_prob_resp(self, expectations, with_prior = False):
        """Estimate log probabilities and responsibilities for each sample.
        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)
        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(expectations, with_prior = with_prior)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _e_step(self, Y = None, with_prior = False):
        if Y is None:
            Y = self.Y
        self.Sigma_w_val = cov_mat_precomp(self.Sigma_w)
        self.Sigma_y_val = cov_mat_precomp(self.Sigma_y)
        inv_sig_w = np.array([temp['inv'] for temp in self.Sigma_w_val])
        inv_sig_y = np.array([temp['inv'] for temp in self.Sigma_y_val])
        # 2) Compute expectations
        w_means = []
        w_covs = []
        alpha = self.alpha
        for n, time in enumerate(self.times):
            Tn = len(Y[n])
            wns = []
            Swns = []
            for i in range(self.n_components):
                concat_Phi = np.concatenate(self.Phi[n], axis=1)
                concat_tmp1 = np.einsum('ji,jk', concat_Phi, inv_sig_y[i], optimize=True)
                concat_tmp1 = concat_tmp1.reshape(Tn, self.w_dim, -1)
                sum_mean = np.dot(inv_sig_w[i], self.mu_w[i]) + np.einsum("ijk,ik", concat_tmp1, Y[n], optimize=True)
                sum_cov = inv_sig_w[i] + np.einsum("ijk,ikm", concat_tmp1, self.Phi[n], optimize=True)
                Swn = np.linalg.inv(sum_cov)
                wn = np.dot(Swn, sum_mean)
                wns.append(wn)
                Swns.append(Swn)
            w_means.append(np.dot(alpha[np.newaxis, :], wns).flatten())
            w_covs.append(np.einsum('ij,jkl->kl', alpha.reshape(1, -1), Swns, optimize=True))
        expections = {'w_means': np.array(w_means), 'w_covs': np.array(w_covs)}
        log_prob_norm, log_resp = self._estimate_log_prob_resp(expections, with_prior = with_prior)
        expections['log_resp'] = log_resp
        return np.mean(log_prob_norm), expections

    def _m_step(self, expectations):
        w_covs = expectations['w_covs']
        w_means = expectations['w_means']
        log_resp = expectations['log_resp']
        resp = np.exp(log_resp)
        N = len(self.times)
        mu_w, Sigma_w, Sigma_y, alpha = [], [], [], []
        for i in range(self.n_components):
            #1) Optimize mu_w
            mc = np.sum(resp[:,i]) + 10 * np.finfo(resp.dtype).eps
            alpha_i = mc
            wn_sum = np.dot(resp[:,i], w_means)
            if self.prior_mu_w is not None:
                mu_w_i = (wn_sum + self.prior_mu_w['k0']*self.prior_mu_w['m0'])/(mc + self.prior_mu_w['k0'])
            else:
                mu_w_i = (wn_sum) / mc

            #2) Optimize Sigma_w
            diff_w = np.array([x - mu_w_i for x in w_means])
            n_var = lambda X: sum([np.outer(x, x) for x in X])
            if self.no_Sw:
                sw_sum = force_sym(np.dot(resp[:, i] * diff_w.T, diff_w))
            else:
                sw_sum = force_sym(np.einsum('ij, jkl->kl', resp[:,i].reshape(1,-1), w_covs) + np.dot(resp[:, i] * diff_w.T, diff_w))
            sw_sum = make_block_diag(sw_sum, self.n_rfs)
            __Sigma_w_mle_i = sw_sum / mc  # Maximum likelyhood estimate for Sigma_w
            self.__Sigma_w_mle[i] = __Sigma_w_mle_i

            if self.prior_Sigma_w is not None:
                v0 = self.prior_Sigma_w['v']
                D = int(np.shape(self.Sigma_w[i])[0] / self.n_rfs)
                if 'mean_cov_mle' in self.prior_Sigma_w:
                    S0 = self.prior_Sigma_w['mean_cov_mle'](__Sigma_w_mle_i) * (v0 + D + 1)
                else:
                    S0 = self.prior_Sigma_w['invS0']
                Sigma_w_i = (S0 + sw_sum) / (mc + v0 + D + 1)
            else:
                Sigma_w_i = __Sigma_w_mle_i

            # 3) Optimize Sigma_y
            diff_y = []
            uncert_w_y = []
            for n in range(N):
                for t in range(len(self.times[n])):
                    # diff_y.append((self.Y[n][t] - np.dot(self.Phi[n][t], w_means[n])) * resp[n][i])
                    # uncert_w_y.append(np.dot(np.dot(self.Phi[n][t], w_covs[n]), self.Phi[n][t].T) * resp[n][i])
                    diff_y.append(self.Y[n][t] - np.dot(self.Phi[n][t], w_means[n]))
                    uncert_w_y.append(np.dot(np.dot(self.Phi[n][t], w_covs[n]), self.Phi[n][t].T))
            if self.no_Sw:
                Sigma_y_i = (n_var(diff_y)) / len(diff_y)
            else:
                Sigma_y_i = (n_var(diff_y) + sum(uncert_w_y)) / len(diff_y)

            mu_w.append(mu_w_i)
            Sigma_w.append(Sigma_w_i)
            Sigma_y.append(Sigma_y_i)
            alpha.append(alpha_i)

        # 4) Update
        self.mu_w = np.array(mu_w)
        self.alpha = np.array(alpha) / np.sum(alpha)
        self.Sigma_w = np.array([force_sym(make_block_diag(s, self.n_rfs) + self.reg_covar * np.identity(s.shape[0])) for s in Sigma_w])
        if self.covariance_type == 'diag':
            self.Sigma_w = np.array([np.diag(np.diag(Sigma_w_i)) for Sigma_w_i in self.Sigma_w])
        elif self.covariance_type == 'block_diag':
            self.Sigma_w = np.array(
                [make_block_diag(s, self.n_rfs) for s in self.Sigma_w])
        self.Sigma_y = [force_sym(make_block_diag(s, self.n_rfs)) for s in Sigma_y]
        if self.diag_sy:
            self.Sigma_y = [np.diag(np.diag(s)) for s in Sigma_y]
        else:
            self.Sigma_y = [force_sym(s) for s in Sigma_y]

    def _get_parameters(self):
        return (
            self.alpha,
            self.mu_w,
            self.Sigma_w,
            self.Sigma_y
        )

    def _set_parameters(self, params):
        (
            self.alpha,
            self.mu_w,
            self.Sigma_w,
            self.Sigma_y
        ) = params

    def train(self, times, data, qd = None, max_iter = 30, diag_sy = True, no_Sw = False, prior_Sigma_w = None, prior_mu_w = None, print_lowerbound = False, n_init = 1):
        self.diag_sy = diag_sy
        self.no_Sw = no_Sw
        self.prior_Sigma_w = prior_Sigma_w
        self.prior_mu_w = prior_mu_w
        self.print_lowerbound = print_lowerbound
        self.converged = False
        max_lower_bound = -np.inf
        self.data = data
        self.Y = np.array(self.get_Y(times, data, qd))
        self.Phi = np.array(self.get_Phi(times))
        self.times = np.array(times)
        for init in range(n_init):
            self.initialize()
            self.__Sigma_w_mle = self.Sigma_w
            lower_bound = -np.inf
            for it in range(max_iter):
                prev_lower_bound = lower_bound
                log_prob_norm, expectations = self._e_step()
                self._m_step( expectations)
                lower_bound = log_prob_norm
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
        _, expectations = self._e_step()
        log_resp = expectations['log_resp']
        self.rho = log_resp.argmax(axis = 1)
        print(f'The best iteration is {self.n_iter}')
        print(self.rho)

    def refine(self, max_iter, tol = 1e1):
        self.covariance_type = 'block_diag'
        _, expectations = self._e_step()
        log_resp = expectations['log_resp']
        for i in range(self.n_components):
            if i not in self.rho:
                continue
            index_component = np.where(log_resp.argmax(axis = 1) == i)[0]
            data_component = np.array(self.data)[index_component]
            times_component = np.array(self.times)[index_component]
            model_component = promp_gaussian.FullProMP(self.basis, self.n_dims, n_rfs = self.n_rfs, tol = tol)
            model_component.train(times_component, data_component, prior_Sigma_w = self.prior_Sigma_w, prior_mu_w = self.prior_mu_w, print_lowerbound = self.print_lowerbound, max_iter = max_iter)
            self.mu_w[i], self.Sigma_w[i], self.Sigma_y[i] = model_component.mu_w, model_component.Sigma_w, model_component.Sigma_y

    def _n_parameters(self):
        '''Return the number of free parameters in the model'''
        mean_params = self.n_components * self.w_dim
        if self.covariance_type == 'block_diag':
            cov_params = self.n_components * self.block_dim * (self.block_dim + 1) / 2
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * self.w_dim
        return int(cov_params + mean_params + self.n_components - 1)

    def marginal_w(self, time, mode_selected):
        phi_n = self.get_Phi([time])[0]
        means, covs= [], []
        for phi_t in phi_n:
            means.append(np.dot(phi_t, self.mu_w[mode_selected]))
            covs.append(np.dot(np.dot(phi_t,self.Sigma_w[mode_selected]),phi_t.T))
        return np.array(means), np.array(covs)

    def select_mode(self, time, initial_pos):
        phi_n = self.get_Phi([time])[0]
        phi_n_init = phi_n[0]
        llhs = []
        for i in range(self.n_components):
            means_component_init = np.dot(phi_n_init, self.mu_w[i])
            covs_component_init = np.dot(np.dot(phi_n_init, self.Sigma_w[i]), phi_n_init.T)
            llh = multivariate_normal.logpdf(initial_pos, mean = means_component_init, cov = covs_component_init)
            llhs.append(llh)
        mode_selected = np.argmax(llhs)
        return mode_selected
