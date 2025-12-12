from variational_inference_utils import *
from scipy.special import polygamma, gammaln
from scipy.stats import entropy
import math


class VI_sLDA_M_Step:
    '''
    The default mode for the variational M step is minibatch natural gradient procedure
    '''
    def __init__(self, K, bow, y, alpha, xi, eta, delta, Lambda, gamma, phi, corpus_size, rho=None):
        self.K = K # number of topics
        self.bow = bow # Bag of words: dictionary of list of word indices, with length D
        self.doc_len = {d:len(v) for d,v in bow.items()} # number of words within each document
        self.D = len(self.bow) # batch_size: number of documents in the minibatch
        self.y = y # D-dimensional vector
        self.alpha = alpha # K-dimensional vector
        self.new_alpha = None
        self.xi = xi # V-dimensional vector
        self.new_xi = None
        self.eta = eta # K-dimensional vector
        self.new_eta = None
        self.delta = delta # scalar
        self.new_delta = None
        self.Lambda = Lambda # global variational parameter Lambda (size: K x V)
        self.new_Lambda = None
        self.gamma = gamma # local variational parameters gamma from E step (size: D x K)
        self.phi = phi # local variational parameters phi from E step (dictionary: for each document, size is N_d x K)
        self.phi_bar = np.vstack([self.phi[d].mean(axis=0) for d in range(self.D)]) # average phi of each document (size: D x K)
        phi_minus_j = {d:(self.phi[d].sum(axis=0) - self.phi[d]) for d in range(self.D)} # dictionary: for each document, size is N_d x K
        self.expect_x_x_t = np.zeros(shape=(K,K)) # size: K x K (only dependent on local variational parameter phi)
        for d in range(self.D): # Eq (29) & (35) of the sLDA paper
            N_d = self.doc_len[d]
            self.expect_x_x_t += 1/N_d**2 * (self.phi[d].T @ phi_minus_j[d]) # first term of E[Z @ Z^T]
            self.expect_x_x_t += 1/N_d**2 * np.diag(self.phi[d].sum(axis=0)) # second term of E[Z @ Z^T]
        self.rho = rho
        self.corpus_size = corpus_size
        self.scale_factor = corpus_size / self.D
        
    def update_Lambda(self, batch = False):
        # update rule for the global variational parameter Lambda: See Eq (33) of the SVI paper
        # depends on local variational parameter phi from the E-step
        Lambda_hat = np.zeros_like(self.Lambda) # natural gradient of ELBO w.r.t the variational distribution q(beta | Lambda)
        for d in range(self.D):
            for wi,v in enumerate(self.bow[d]): # wi is the wi^th word in the d^th document, v is this word's index in the Vocabulary
                Lambda_hat[:,v] += self.phi[d][wi,:] # self.phi[d][wi,:] is in fact the variational posterior distribution of topics for the wi^th word in the d^th document
        Lambda_hat = self.scale_factor * Lambda_hat # scale based on minibatch size
        Lambda_hat += self.xi # same for each variational topic distribution parameter
        if batch == False:
            self.new_Lambda = stochastic_variational_update(self.Lambda, Lambda_hat, self.rho)
        else:
            self.Lambda = Lambda_hat
        
    def update_alpha(self, batch = False):
        # update rule for the global hyperparameter alpha: (See section A.2 and A.4.2 of the LDA paper)
        # depends on local variational parameter gamma from the E-step
        alpha_sum = np.sum(self.alpha)
        g = self.D * (polygamma(0, alpha_sum) - polygamma(0, self.alpha))
        g += polygamma(0, self.gamma).sum(axis=0) - np.sum(polygamma(0, self.gamma.sum(axis=1))) # gradient of ELBO w.r.t. alpha
        g = self.scale_factor * g # scale based on minibatch size
        h = -self.corpus_size * polygamma(1, self.alpha) # trigamma: diagonal elements
        z = self.corpus_size * polygamma(1, alpha_sum) # trigamma: z in the 1z1^T term
        alpha_hat = linear_time_natural_gradient(g, h, z) # compute (the negative scaled) natural gradient of ELBO w.r.t. p(theta_{1:corpur_size} | alpha)
        if batch == False:
            self.new_alpha = stochastic_hyperparameter_update(self.alpha, alpha_hat, self.rho)
        else: # in the batch VI mode, this corresponds to one iteration of New-Raphson algorithm to update alpha
            self.new_alpha = self.alpha - alpha_hat # not update the value of alpha here because we need both old & new values for checking convergence 

    def update_xi(self, batch=False):
        # update rule for the global hyperparameter xi:
        # depends on global variational parameter Lambda from the previous variational EM iteration
        xi_sum = np.sum(self.xi)
        g = self.K * (polygamma(0, xi_sum) - polygamma(0, self.xi))
        g += polygamma(0, self.Lambda).sum(axis=0) - np.sum(polygamma(0, self.Lambda.sum(axis=1))) # gradient of ELBO w.r.t xi
        h = -self.K * polygamma(1, self.xi)
        z = self.K * polygamma(1, xi_sum)
        xi_hat = linear_time_natural_gradient(g, h, z) # compute the (negative) natural gradient of ELBO w.r.t. p(beta_{1:K} | xi)
        if batch == False:
            self.new_xi = stochastic_hyperparameter_update(self.xi, xi_hat, self.rho)
        else: # in the batch VI mode, this corresponds to one iteration of New-Raphson algorithm to update alpha
            self.new_xi = self.xi - xi_hat # not update the value of xi here because we need both old & new values for checking convergence

    def is_elbo_eta_delta_improved(self, old_elbo, eta_hat, delta_hat):
        # this function check whether an update for (eta, delta) improves the ELBO (the terms that depend on (eta, delta))
        new_delta = self.delta - delta_hat
        if new_delta <= 0:
            return False
        else:
            new_eta = self.eta - eta_hat        
            y_t_y = np.sum(self.y**2)
            phi_bar_times_y = np.dot(self.y, self.phi_bar) # K-dimensional vector
            new_elbo = -self.D/2*np.log(new_delta) - 1/2/(new_delta) * (y_t_y + np.dot(new_eta, np.dot(self.expect_x_x_t, new_eta)) - 2*np.dot(new_eta, phi_bar_times_y))
            return new_elbo > old_elbo
        
    def update_eta_and_delta(self, batch=False):
        # joint update rule for the global hyperparameter (eta, delta) (Gaussian response):
        # depends on the local variational parameter phi from the E-step
        phi_bar_times_y = np.dot(self.y, self.phi_bar) # K-dimensional vector
        expect_x_x_t_times_eta = np.dot(self.expect_x_x_t, self.eta) # K-dimensional vector
        y_t_y = np.sum(self.y**2)
        temp_var = np.dot(self.eta, phi_bar_times_y - expect_x_x_t_times_eta/2) # dot product
        g_eta = (1/self.delta)*(phi_bar_times_y - expect_x_x_t_times_eta) # K-dimensional vector
        g_delta = -self.D/2/self.delta + 1/2/self.delta**2 * (y_t_y - 2*temp_var) # the term y_t_y - 2*temp_var is positive
        g = self.scale_factor * np.hstack([g_eta, np.array([g_delta])]) # gradient is of K+1 dimensional, scale based on minibatch size
        h_11 = -self.expect_x_x_t/self.delta
        h_21 = -g_eta / self.delta # mixed partial derivatives: K-dimensional vector
        h_22 = self.D/2/self.delta**2 - 1/self.delta**3 * (y_t_y - 2*temp_var)
        h = np.zeros(shape=(self.K+1, self.K+1)) # Hessian of L w.r.t (eta, delta)
        h[:self.K, :self.K] = h_11
        h[self.K, self.K] = h_22
        h[self.K, :self.K] = h_21
        h[:self.K, self.K] = h_21
        h = self.scale_factor * h # (scaled) Hessian is of (K+1) x (K+1) dimensional
        h_inv = np.linalg.inv(h) # inverse of the Hessian matrix
        eta_delta_hat = np.hstack([np.ones(self.K), [0.1]]) * (h_inv @ g) # the approximated (negative) natural gradient of ELBO w.r.t p(Y_{1:corpus_size}|eta, delta) pre-multiplied by a constant vector
        # Determine the appropriate step size for the natural gradient step
        accept = (False, False)
        # find the proper step size for each New-Raphson update through trial and error
        step_size = np.sqrt(2)
        while accept != (True, True):
            step_size = step_size / 2
            eta_hat = step_size * eta_delta_hat[:self.K]
            delta_hat = step_size * eta_delta_hat[self.K]
            old_elbo = -self.D/2*np.log(self.delta) - 1/2/(self.delta) * (y_t_y + np.dot(self.eta, np.dot(self.expect_x_x_t, self.eta)) - 2*np.dot(self.eta, phi_bar_times_y))
            current_accept = self.is_elbo_eta_delta_improved(old_elbo, eta_hat, delta_hat)
            accept = (accept[1], current_accept)
        step_size = np.sqrt(2) * step_size
        if batch == False: # natural gradient step in the M step of stochastic VI mode
            updated_eta_delta = stochastic_hyperparameter_update(np.hstack([self.eta, [self.delta]]), step_size * eta_delta_hat, self.rho)
            self.new_eta = updated_eta_delta[:self.K] # K-dimensional vector
            self.new_delta = updated_eta_delta[self.K] # scalar
        else: # Newton-Raphson step in batch VI mode
            self.new_eta = self.eta - step_size * eta_delta_hat[:self.K]
            self.new_delta = self.delta - step_size * eta_delta_hat[self.K]
        
    def run(self, supervised=True, update_alpha_and_xi=True):
        # run one full M-step: update rules for these 4 set of global parameters together form one step in the stochastic (minibatch) natural gradient ascent procedure
        self.update_Lambda()
        if update_alpha_and_xi == True:
            self.update_alpha()
            self.update_xi()
        if supervised == True: # run minibatch sLDA mode if True; run minibatch LDA model if False
            self.update_eta_and_delta()
            return self.new_Lambda, self.new_alpha, self.new_xi, self.new_eta, self.new_delta # output the global parameters to be used in the next iteration of stochastic (minibatch) variational EM
        else:
            if update_alpha_and_xi == True:
                return self.new_Lambda, self.new_alpha, self.new_xi, self.eta, self.delta
            else:
                return self.new_Lambda, self.alpha, self.xi, self.eta, self.delta


class batch_VI_sLDA_M_Step(VI_sLDA_M_Step):
    '''
    This is the variational M step used in batch mode. It's a child class of VI_sLDA_M_Step as many of the methods are the same in batch and stochastic
    (minibatch mode).
    '''
    def __init__(self, K, bow, y, alpha, xi, eta, delta, Lambda, gamma, phi, corpus_size, epsilon, closed_form = True): 
        super().__init__(K, bow, y, alpha, xi, eta, delta, Lambda, gamma, phi, corpus_size)
        self.epsilon = epsilon # stopping criteria for checking the convergence criteria for Newton-Raphson for alpha and xi, and potentially for eta and delta
        self.closed_form = closed_form
        self.elbo = 0 # corpus-level ELBO, which is evaluated at the end of each full M step

    def optimize_Lambda(self):
        # optimize the global variational parameter Lambda in the M step in batch mode VI:
        # it has a closed-form solution
        self.update_Lambda(batch=True) # in botch mode VI, the optimization of xi relies on the optimized values of Lambda
    
    def optimize_alpha(self):
        # run a full Newton-Raphson procedure to optimize alpha in the M step
        change_in_alpha = math.inf
        while change_in_alpha > self.epsilon: # convergence criteria
            self.update_alpha(batch=True)
            change_in_alpha = np.mean(np.abs(self.new_alpha - self.alpha))
            self.alpha = self.new_alpha.copy()

    def optimize_xi(self):
        # run a full Newton-Raphson procedure to optimize xi in the M step
        # Note: dependent on optimized values of Lambda
        change_in_xi = math.inf
        while change_in_xi > self.epsilon: # convergence criteria
            self.update_xi(batch=True)
            change_in_xi = np.mean(np.abs(self.new_alpha - self.alpha))
            self.xi = self.new_xi.copy()

    def optimize_eta_and_delta(self):
        # default method: optimization in terms of eta and delta has a closed-form solution in batch mode VI (Eq (34) & (37) of the SVI paper)
        if self.closed_form == True:
            expect_x_x_t_inv = np.linalg.inv(self.expect_x_x_t)
            phi_bar_times_y = np.dot(self.y, self.phi_bar) # K-dimensional vector
            new_eta = np.dot(expect_x_x_t_inv, phi_bar_times_y)
            self.delta = 1/self.D * (np.sum(self.y**2) - np.dot(phi_bar_times_y, self.eta)) # need to use the optimized value of eta to find the optimized value of delta
            self.eta = new_eta
        else: # mostly for diagnostic purposes
            change_in_eta_delta = math.inf
            while change_in_eta_delta > self.epsilon: # convergence criteria
                self.update_eta_and_delta(batch=True)
                change_in_eta_delta = (np.sum(np.abs(self.new_eta - self.eta)) + np.abs(self.new_delta - self.delta)) / (self.K + 1)
                self.eta = self.new_eta.copy()
                self.delta = self.new_delta.copy()

    def compute_elbo(self):
        '''
        Note: gammaln() computes the natural log of Gamma function in a numerically stable way
        '''
        # the corpus-level ELBO computed at the end of every variational EM iteration, which will be used to determine the stopping time of the entire variational EM
        alpha_sum = np.sum(self.alpha)
        temp_var_1 = polygamma(0, self.gamma).T - polygamma(0, self.gamma.sum(axis=1)) # size: K x D (uses broadcasting)
        self.elbo += self.D * (gammaln(alpha_sum) - np.sum(gammaln(self.alpha))) + np.sum(np.dot(self.alpha-1, temp_var_1)) # E_q[log p(theta_d | alpha)], summing over d
        for d in range(self.D): # E_q[log p(Z_dn | theta_d)] # summing over d and n
            self.elbo += np.dot(self.phi[d].sum(axis=0), temp_var_1[:,d]) # Note: could sum phi_nk over n first as the other factor doesn't depend on n
        xi_sum = np.sum(self.xi)
        temp_var_2 = polygamma(0, self.Lambda).T - polygamma(0, self.Lambda.sum(axis=1)) # size: V x K (doesn't depend on d)
        for d in range(self.D): # E_q[log p(w_dn | Z_dn, beta_{1:K}], summing over d and n
            for wi,v in enumerate(self.bow[d]): # wi is the wi^th word in the d^th document, v is the word's index in the Vocabulary
                # This is looping through n = 1, 2, ..., N_d
                self.elbo += np.dot(self.phi[d][wi,:], temp_var_2[v,:]) # dot product of two K-dimensional vectors
        y_t_y = np.sum(self.y**2)
        phi_bar_times_y = np.dot(self.y, self.phi_bar) # K-dimensional vector
        self.elbo += -self.D/2*np.log(self.delta) - 1/2/self.delta * (y_t_y + np.dot(self.eta, np.dot(self.expect_x_x_t, self.eta)) - 2*np.dot(self.eta, phi_bar_times_y)) # E_q[p(y_d | Z_d, eta, delta)], summing over d
        self.elbo += self.K * (gammaln(xi_sum) - np.sum(gammaln(self.xi))) + np.sum(np.dot(self.xi-1, temp_var_2)) # E_q[p(beta_k | lambda_k)], summing over k
        temp_var_3 = 0
        for d in range(self.D):
            temp_var_3 += np.dot(self.gamma[d,:]-1, temp_var_1[:,d])
        self.elbo -= np.sum(gammaln(self.gamma.sum(axis=1))) - np.sum(gammaln(self.gamma).sum(axis=1)) + temp_var_3 # H(q(theta_d | gamma_d)), summing over d
        temp_var_4 = 0
        for d in range(self.D):
            for n in range(self.doc_len[d]):
                temp_var_4 -= entropy(self.phi[d][n,:], base=np.exp(1)) # the entropy function could handle 0 * log(0) nicely
        self.elbo -= temp_var_4 # H(q(Z_dn | phi_dn)), summing over d and n
        temp_var_5 = 0
        for k in range(self.K):
            temp_var_5 += np.dot(self.Lambda[k,:]-1, temp_var_2[:,k])
        self.elbo -= np.sum(gammaln(self.Lambda.sum(axis=1))) - np.sum(gammaln(self.Lambda).sum(axis=1)) + temp_var_5 # H(q(beta_k | lambda_k)), summing over k
        
    def run(self, supervised = True, optimize_alpha_and_xi = True):
        # override the .run() method in the parent Class
        # run the full M step
        self.optimize_Lambda()
        if optimize_alpha_and_xi == True: # default option is to optimize the global hyperparameters alpha and xi in M step; if False, then we just use initial values of those parameters throughout
            self.optimize_alpha()
            self.optimize_xi()
        if supervised == True: # default option is to run M step of sLDA; if supervised == False, then run M step of LDA. 
            self.optimize_eta_and_delta()
        self.compute_elbo()
        return self.Lambda, self.alpha, self.xi, self.eta, self.delta, self.elbo