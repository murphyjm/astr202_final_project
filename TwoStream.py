import numpy as np
from tqdm import tqdm
from math import isclose

class TwoStream:
    '''
    An object for doing numerical radiative transfer on a TwoStream slab.

    Attributes
    ---------------
    max_iters     (int): Maximum number of iterations to use in iterate().
    z         (ndarray): Spatial grid. Sampled in log space, so that the region near z=0
                            gets a sufficient amount of sampling, since that's where the quantities will
                            vary greatly.
    sample_freq   (int): Frequency (1/iterations) at which to record the source function, S.
    epsilon     (float): Photon destruction probability.
    B           (float): Incident intensity. Leave as 1, since we plot in units of B.
    alpha       (float): Extinction coefficient as a function of z.
    converged    (bool): Did S converge to a stable value during the iteration loop?
    tau       (ndarray): Optical depth, as computed from alpha and z.
    delta_tau (ndarray): Differences of consecutive elements of tau. delta_tau_i = tau_i - tau_{i+1}.
    alpha, beta, gamma_plus/minus (ndarray): Coefficients for integrating the updated intenties as a
                                                weighted sum of the source function.
    I_plus    (ndarray): Upward intensity. BC: I_plus = B at z = 0.
    I_minus   (ndarray): Downward intensity. BC: I_minus = 0 at z = 1.
    J         (ndarray): Average intensity: J = 0.5 * (I_plus + I_minus).
    S         (ndarray): Source function.
    S_series     (list): List of source functions, sampled every sample_freq iterations.
    '''

    def __init__(self, max_iters=100, grid_num=int(1e5), sample_freq=10, epsilon=0.1, B=1.):
        '''
        Initialize the TwoStream object.

        Args
        ---------------
        max_iters   (int): Maximum number of iterations.
        grid_num    (int): Number of points to include in the grid.
        sample_freq (int): Frequency (1/iterations) at which to record S.
        epsilon   (float): Photon destruction probability.
        B         (float): Incident intensity at the bottom (z=0) of the slab.
                           Leave this as 1, since we'll plot everything in units of B.
        '''
        # Hyperparameters
        self.max_iters = max_iters
        # self.z = np.logspace(-6, 0, num=grid_num) # Sample in log space to give us sufficient resolution near z=0.
        self.z = np.linspace(0., 1.0, num=grid_num)
        self.sample_freq = sample_freq
        self.epsilon = epsilon
        self.B = B
        self.alpha = np.power(10., 5 - 6*self.z) # Extinction coefficient

        # Convergence met during iteration?
        self.converged = False

        # Integrated from dT = alpha dz / mu. Integration constant shouldn't matter because we actually only use delta_tau.
        self.tau = np.sqrt(3) * (np.power(4., 2 - 3*self.z) * np.power(5., 5 - 6*self.z) / (3 * np.log(10))) # Optical depth
        self.delta_tau = -1 * np.ediff1d(self.tau) # -1 factor so that delta_tau = tau_i - tau_{i+1}.

        # Initial conditions
        self.calculate_coeffs()       # Initialize the alpha, beta, gamma coefficients
        self.set_initial_conditions() # Set initial conditions for I_minus, I_plus, J, and S

    def calculate_coeffs(self):
        '''
        Set the coefficients for the alpha, beta, and gamma arrays.
        '''

        # Convenience variables
        delta_tau = np.copy(self.delta_tau[:-1])
        delta_tau_iplus1 = np.copy(self.delta_tau[1:])
        delta_tau_sum = delta_tau + delta_tau_iplus1

        # Set the I_plus coefficients
        e_plus_0 = 1 - np.exp(-1 * delta_tau)
        e_plus_1 = delta_tau - e_plus_0
        e_plus_2 = np.square(delta_tau) - 2 * e_plus_1
        self.alpha_plus = (e_plus_2 - delta_tau * e_plus_1) / (delta_tau_iplus1 * delta_tau_sum)
        self.beta_plus  = (delta_tau_sum * e_plus_1 - e_plus_2) / (delta_tau_iplus1 * delta_tau)
        self.gamma_plus = e_plus_0 + (e_plus_2 - (delta_tau_iplus1 + 2 * delta_tau) * e_plus_1) / (delta_tau * delta_tau_sum)

        # Set the I_minus coefficients
        e_minus_0 = 1 - np.exp(-1 * delta_tau_iplus1)
        e_minus_1 = delta_tau_iplus1 - e_minus_0
        e_minus_2 = np.square(delta_tau_iplus1) - 2 * e_minus_1
        self.alpha_minus = e_minus_0 + (e_minus_2 - (delta_tau + 2 * delta_tau_iplus1) * e_minus_1) / (delta_tau_iplus1 * delta_tau_sum)
        self.beta_minus  = (delta_tau_sum * e_minus_1 - e_minus_2) / (delta_tau * delta_tau_iplus1)
        self.gamma_minus = (e_minus_2 - delta_tau_iplus1 * e_minus_1) / (delta_tau * delta_tau_sum)

    def set_initial_conditions(self):
        '''
        Set the initial conditions for I_minus and I_plus (and therefore J and S).
        '''
        # Initialize I_plus and I_minus as (arbitrary) smooth functions that satisfy
        # the boundary conditions, but these shouldn't espcially matter
        self.I_plus  = self.B - self.z * np.exp(-0.5 * self.z)
        self.I_minus = np.exp(-1) - self.z * np.exp(-1*self.z)
        self.J       = self.get_J()
        self.S       = self.get_S()

    def iterate(self, converge_tol=1e-6, break_if_converged=False):
        '''
        Main loop for calculating the iterative solution.

        Args
        ---------------
        converge_tol       (float, default=1e-6): Absolute tolerance for meeting
                                                  convergence criterion.
                                                  See convergence_test() for details.
        break_if_converged (bool, default=False): Whether or not to exit the
                                                  loop if we reach the convergence
                                                  criterion. Otherwise, iterate until
                                                  max_iters is met.

        Returns
        ---------------
        output (dict): A dictionary containing relevant output.
        '''
        S_series = []      # Time series of source function
        converged_i = None # Iteration number at which convergence was reached (if at all)

        # Main loop
        for i in tqdm(range(self.max_iters)):
            self.I_minus = self.get_I_minus()
            self.I_plus = self.get_I_plus()
            self.J   = self.get_J()
            self.S   = self.get_S()

            if i == 0 or np.mod(i + 1, self.sample_freq) == 0:
                S_series.append(self.S)

            if i >= self.sample_freq * 2 and self.convergence_test(S_series[-1], S_series[-2], converge_tol):
                self.converged = True

                # Record the first iteration when we reached convergence
                if converged_i is None:
                    converged_i = i

                if break_if_converged:
                    break

        # Print statement regarding convergence and early exits
        if not self.converged:
            print("Warning, source function approximation not converged within tolerance level.")
        else:
            print("Convergence criterion met within tolerance level of {:.2e}.".format(converge_tol))
            if break_if_converged:
                print("Exited early after {} iterations...".format(converged_i))
            else:
                print("Convergence first reached after {} iterations.".format(converged_i))

        self.S_series = S_series

        # A summary of the output
        output = {
        "z":self.z,
        "tau":self.tau,
        "I_minus":self.I_minus,
        "I_plus":self.I_plus,
        "J":self.J,
        "S":self.S,
        "S_series":self.S_series,
        }
        return output

    def get_I_minus(self):
        '''
        Calculate the approximation for the downward stream.

        Returns
        ---------------
        i_minus (ndarray): An updated version of the downward stream.
        '''
        i_minus = np.copy(self.I_minus)

        i_minus[-1] = 0. # Boundary condition
        i_minus[-2] = 0.5 * (i_minus[-3] + i_minus[-1]) # Fencepost problem, just take the average of the neighbors.

        for i in range(len(i_minus)-3, 0, -1): # Step through in reverse, starting at z = 1.
            first_term = i_minus[i+1] * np.exp(-1*self.delta_tau[i+1])
            s_weighted_sum = self.gamma_minus[i] * self.S[i-1] + self.beta_minus[i] * self.S[i] + self.alpha_minus[i] * self.S[i+1]
            i_minus[i] = first_term + s_weighted_sum

        # Fencepost problem at z = 0. Just copy the neighboring element.
        i_minus[0] = i_minus[1]

        return i_minus

    def get_I_plus(self):
        '''
        Calculate the approximation for the upward stream.

        Returns
        ---------------
        i_plus (ndarray): An updated version of the upward stream.
        '''
        i_plus = np.copy(self.I_plus)

        i_plus[0] = self.B # Boundary condition

        for i in range(1, len(i_plus)-2):
            first_term = i_plus[i-1] * np.exp(-1 * self.delta_tau[i-1])
            s_weighted_sum = self.gamma_plus[i] * self.S[i-1] + self.beta_plus[i] * self.S[i] + self.alpha_plus[i] * self.S[i+1]
            i_plus[i] = first_term + s_weighted_sum

        # Fencepost problem, just set the s_weighted_sum term to zero.
        i_plus[-2] = i_plus[-3] * np.exp(-1 * self.delta_tau[-1])

        # Fencepost problem at z = 1. Just copy the preceeding element.
        i_plus[-1] = i_plus[-2]

        return i_plus

    def get_J(self):
        '''
        Return the mean intensity.
        Call get_I_minus() and get_I_plus() before calling this (or at least set
        their initial conditions first).

        Returns
        ---------------
        (ndarray): The average intensity.
        '''
        return 0.5 * (self.I_minus + self.I_plus)

    def get_S(self):
        '''
        Return the source function. Call get_J() before calling this.

        Returns
        ---------------
        (ndarray): The source function.
        '''
        return self.epsilon * self.B + (1 - self.epsilon) * self.J

    def convergence_test(self, S_new, S_old, tol):
        '''
        Check for convergence using the L2 norm.

        Returns
        ---------------
        (bool): Whether or not the L2 norm of the difference of two calculations of
                the source function is within tol of 0.
        '''
        n = len(S_new)
        norm = np.linalg.norm(S_new - S_old) / n # Normalize the tolerance for the number of grid points
        return isclose(0.0, norm, abs_tol=tol)
