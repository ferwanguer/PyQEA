import numpy as np
import time
import os
import sys

class QuantumEvAlgorithm:
    """This object encapsules the necessary methods to make a Quantum-Inspired
    optimization algorithm. For a thorough description of the algorithm please visit:
    URL. FWG"""

    def __init__(self, f, n_dims, upper_bound, lower_bound, integral_id, sigma_scaler=1.00001, mu_scaler=100, elitist_level=4,
    error_ev = 1 ,ros_flag = False, saving_interval = 500, restrictions = []):
        """The QuantumEvAlgorithm class admits a (scalar) function to be optimized. The function
        must be able to generate multiple outputs for multiple inputs of shape (n_samples,n_dimensions).
        The n_dims attribute is to be placed as an input of the class"""
        self.cost_function = f
        self.n_dims = n_dims
        self.sigma_scaler = sigma_scaler
        self.mu_scaler = mu_scaler
        self.elitist_level = elitist_level
        self.ros_flag = ros_flag
        self.error = error_ev
        self.saving_interval = saving_interval
        self.upper = upper_bound
        self.lower = lower_bound
        self.integral_id = integral_id
        self.restrictions = restrictions
        
        assert (np.array(n_dims) == self.lower.shape and np.array(n_dims) == self.upper.shape), f"The dimensions\
of upper and lower bounds do not coincide with the dimensionality of the problem. n_dims = {self.n_dims}\
 vs lower bound = {self.lower.shape} and upper_bound = {self.upper.shape} "
    def quantum_individual_init(self):
        """Creates a Quantum individual of n_dims features. For each feature mu and sigma are created
         (normal distribution).
         First row: mean (mu)
         Second row: std deviation (sigma)
         In this case, the mu creation is random. The initialization of the std deviation is done so that
         a significant part of the domain is covered. (-32<x_i<32) Domain information must be an input of the class.
         (PENDING)"""
        # First row: mean (mu)
        # Second row: std deviation (sigma)

        # np.random.seed(4)
        Q = self.lower + self.upper * np.random.rand(2, self.n_dims)
        Q[1, :] =  (self.upper - self.lower) * np.ones(self.n_dims)

        self.best_of_best = Q[0:1, :]  # Initial definition of best_of_best
        # print(Q)
        return Q

    def quantum_sampling(self, Q, n_samples):
        """This method generates n_samples from Q (each sample feature is generated with its correspondent
        mu_i and sigma_i)"""
       
        if self.integral_id.any() == True:   
            normal_samples = np.random.normal(Q[0, :], Q[1, :], size=(n_samples, self.n_dims))
            mask =  np.tile(self.integral_id, (n_samples,1))
            samples = np.minimum(np.maximum(normal_samples,self.lower),self.upper)
            np.place(samples, mask, np.round(samples[mask]))
        else:
            samples = np.minimum(np.maximum(np.random.normal(Q[0, :], Q[1, :], size=(n_samples, self.n_dims)),self.lower),self.upper)
        
        return samples

    def restricted_quantum_sampling(self, Q, n_samples):
        """This method generates n_samples from Q (each sample feature is generated with its correspondent
        mu_i and sigma_i)"""
        samples = self.quantum_sampling(Q, n_samples)

        valid = np.full(n_samples,True)

        for h in self.restrictions:
            valid = valid * h(samples)>0
        samples = samples[valid,:]
        
        
        while (n_samples-samples.shape[0] > 0):
            new_samples = self.quantum_sampling(Q, n_samples - samples.shape[0])
            
            valid_s = np.full(n_samples - samples.shape[0],True)
            for h in self.restrictions:
                valid_s = valid_s * h(new_samples)>0
            samples = np.vstack((samples,new_samples[valid_s,:]))    



       
            
        return samples



    def elitist_sample_evaluation(self, samples):
        """This function is analog to the previous one. Instead of choosing the best, it computes the mean of the
        0 best samples."""
        cost = self.cost_function(samples)
        sort_order = np.argsort(cost, axis=0)
        elitist_level = self.elitist_level
        best_performing_sample = np.mean(samples[sort_order[0:elitist_level]], axis=0)[None]

        return best_performing_sample




    def quantum_update(self, Q, best_performing_sample):
        """This method updates the Quantum individual with the criteria explained in: URL (PENDING).
        The update mainly depends in two hyper-parameters as defined below:

        scaling: It controls the transformation of mu_(j+1)
        sigma_scaler: It controls the transformation of sigma_(j+1).

        It is PENDING to put them as inputs to the class (Global parameters of the algorithm)"""

        mu = Q[0:1, :]
        sigma = Q[1:2, :]

        scaling = self.mu_scaler
        mu_delta = best_performing_sample - mu
        mu_delta_2 = self.best_of_best - mu

        updated_mu = mu + (mu_delta + mu_delta_2) / scaling

        sigma_decider = np.abs(mu_delta) / sigma
        #print(sigma_decider)
        sigma_scaler = self.sigma_scaler

        updated_sigma = (sigma_decider < 1) * sigma / sigma_scaler + (sigma_decider > 1) * sigma * sigma_scaler
        if self.cost_function(updated_mu)>10 and self.ros_flag:
            condition = (updated_sigma < 0.001) * (sigma_decider < 1)
            updated_sigma[condition] = updated_sigma[condition] * sigma_scaler


        Q[0:1] = updated_mu
        Q[1:2] = updated_sigma

        return Q

    def progress(self, count, total, status='Processing'):
        bar_len = 30
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()

    def training(self, N_iterations=100000, sample_size=5,
                 save = False, filename = 'testing_evl.npz'):

        assert(sample_size > self.elitist_level), "Sample size must be greater than elitist level, byF"
        j = 0


        Q = self.quantum_individual_init()
         

        Q_history = np.zeros((1 + int(N_iterations / self.saving_interval), 2, self.n_dims))
       
        best_performer_marker = np.zeros((1+int(N_iterations / self.saving_interval), 1))
        function_evaluations = np.zeros(1+int(N_iterations / self.saving_interval))

        print('Beginning of the iteration process')
        beginning = time.time()
        for i in range(N_iterations+1):

            # adapted_sample_size = int(sample_size * (1 + sample_increaser_factor * (i / N_iterations)))
            if self.restrictions:
                samples = self.restricted_quantum_sampling(Q, sample_size)
            else:
                samples = self.quantum_sampling(Q, sample_size)
            
            best_performer = self.elitist_sample_evaluation(samples)
            
            if self.cost_function(best_performer) < self.cost_function(self.best_of_best):
                self.best_of_best = best_performer

            Q = self.quantum_update(Q, best_performer)

            if np.mod(i, self.saving_interval) == 0:
                Q_history[j, :, :] = Q
                output = self.cost_function(best_performer)
                best_performer_marker[j, :] = output
                function_evaluations[j] = i * (sample_size)# + (sample_increaser_factor * i) / 2)
                j += 1
                

            if np.mod(i, 50) == 0:
                # print(f'Progress {100*i/N_iterations:.2f}%, Best cost = {output}')
                self.progress(i,N_iterations,f'Best cost = {self.cost_function(best_performer)}')

            if self.cost_function(best_performer) <= 0.0001 and False:
                print(f'\n\n|| Min detected, value = {self.cost_function(best_performer)}||')
                break

        end = time.time()

        results = {
            "time": end - beginning,
            "cost": self.cost_function(best_performer),
            "min" : best_performer  
        }

        print(f' \n \n Elapsed time = {end - beginning} seconds')
                
        #print(f' min is  = {self.cost_function(best_performer)}')

        # print(f'The min is IN = {best_performer} \n \n ')

        results_path = 'Results'
        if save:
            print(' Saving results')
            np.savez(os.path.join(results_path, filename), best_performer_marker, Q_history, function_evaluations,
                    cost_h = best_performer_marker,
                    pos_history = Q_history, time=function_evaluations)
        print(100*'-')
        return results

     

