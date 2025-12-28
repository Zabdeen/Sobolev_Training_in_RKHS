"""
Sobolev Training in RKHS - Experimental Reproduction
Based on "A Theoretical Analysis of Using Gradient Data for Sobolev Training in RKHS"
by Zain ul Abdeen et al., IFAC PapersOnLine 56-2 (2023) 3417-3422
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class SobolevTraining:
    """Implementation of Sobolev Training with Gaussian Kernel"""
    
    def __init__(self, gamma=1.0, lambda_reg=0.01, beta=1.0):
        """
        Parameters:
        -----------
        gamma : float
            Gaussian kernel bandwidth
        lambda_reg : float
            Regularization parameter
        beta : float
            Weight for gradient data (0 for classical, >0 for Sobolev)
        """
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.beta = beta
        self.alpha = None
        self.X_train = None
        
    def gaussian_kernel(self, X1, X2):
        """Compute Gaussian (RBF) kernel matrix"""
        dist_sq = cdist(X1, X2, metric='sqeuclidean')
        return np.exp(-dist_sq / (2 * self.gamma**2))
    
    def gradient_kernel(self, X1, X2, dim=0):
        """Compute gradient of Gaussian kernel with respect to dimension dim"""
        K = self.gaussian_kernel(X1, X2)
        diff = X1[:, dim:dim+1] - X2[:, dim:dim+1].T
        return -(diff / self.gamma**2) * K
    
    def fit(self, X_train, y_train, grad_train=None):
        """
        Train the model
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training input data
        y_train : array-like, shape (n_samples,)
            Training output data
        grad_train : array-like, shape (n_samples, n_features)
            Training gradient data (only used if beta > 0)
        """
        self.X_train = np.array(X_train)
        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)
        
        y_train = np.array(y_train).flatten()
        n, d = self.X_train.shape
        
        # Compute kernel matrix
        K = self.gaussian_kernel(self.X_train, self.X_train)
        
        if self.beta == 0 or grad_train is None:
            # Classical training: only use function values
            self.alpha = np.linalg.solve(K + self.lambda_reg * np.eye(n), y_train)
        else:
            # Sobolev training: use both function values and gradients
            grad_train = np.array(grad_train)
            if grad_train.ndim == 1:
                grad_train = grad_train.reshape(-1, 1)
            
            # Augmented formulation
            K_aug = K.copy()
            y_aug = y_train.copy()
            
            for j in range(d):
                K_grad_j = self.gradient_kernel(self.X_train, self.X_train, j)
                K_aug += self.beta * K_grad_j.T @ K_grad_j
                y_aug += self.beta * K_grad_j.T @ grad_train[:, j]
            
            self.alpha = np.linalg.solve(K_aug + self.lambda_reg * np.eye(n), y_aug)
        
        return self
    
    def predict(self, X_test):
        """Predict outputs for test data"""
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        K_test = self.gaussian_kernel(X_test, self.X_train)
        return K_test @ self.alpha


def generate_styblinski_tang_data(n, L):
    """
    Generate data based on scaled Styblinski-Tang function
    
    Parameters:
    -----------
    n : int
        Number of samples
    L : float
        Lipschitz constant (controls scaling)
    
    Returns:
    --------
    X : array, shape (n, 1)
        Input samples in [-1, 1]
    y : array, shape (n,)
        Function values
    grad_y : array, shape (n, 1)
        Gradient values
    """
    X = 2 * np.random.rand(n, 1) - 1  # Uniform in [-1, 1]
    
    # Styblinski-Tang function: f(x) = (x^4 - 16*x^2 + 5*x) / 2
    # Scale to control Lipschitz constant
    scale = L / 50.0
    
    y = scale * (X**4 - 16*X**2 + 5*X).flatten() / 2
    grad_y = scale * (4*X**3 - 32*X + 5) / 2
    
    # Add small noise
    y += 0.01 * np.random.randn(n)
    grad_y += 0.01 * np.random.randn(n, 1)
    
    return X, y, grad_y


def experiment_1_lipschitz_effect():
    """Experiment 1: Effect of Lipschitz Constant on RMSE"""
    print("Running Experiment 1: Effect of Lipschitz Constant on RMSE")
    
    # Parameters
    n_train = 30
    n_test = 20
    n_runs = 100
    lipschitz_values = np.arange(10, 510, 10)
    gamma = 1.0
    lambda_reg = 0.01
    
    # Storage for results
    rmse_classical = np.zeros((len(lipschitz_values), n_runs))
    rmse_sobolev = np.zeros((len(lipschitz_values), n_runs))
    
    for L_idx, L in enumerate(lipschitz_values):
        if L_idx % 5 == 0:
            print(f"  Processing Lipschitz constant: {L}")
        
        for run in range(n_runs):
            # Generate data
            X_train, y_train, grad_train = generate_styblinski_tang_data(n_train, L)
            X_test, y_test, _ = generate_styblinski_tang_data(n_test, L)
            
            # Classical training (beta = 0)
            model_classical = SobolevTraining(gamma=gamma, lambda_reg=lambda_reg, beta=0)
            model_classical.fit(X_train, y_train)
            y_pred_classical = model_classical.predict(X_test)
            rmse_classical[L_idx, run] = np.sqrt(np.mean((y_test - y_pred_classical)**2))
            
            # Sobolev training (beta = 1)
            model_sobolev = SobolevTraining(gamma=gamma, lambda_reg=lambda_reg, beta=1.0)
            model_sobolev.fit(X_train, y_train, grad_train)
            y_pred_sobolev = model_sobolev.predict(X_test)
            rmse_sobolev[L_idx, run] = np.sqrt(np.mean((y_test - y_pred_sobolev)**2))
    
    # Plot results
    plt.figure(figsize=(10, 7))
    
    mean_classical = np.mean(rmse_classical, axis=1)
    std_classical = np.std(rmse_classical, axis=1)
    mean_sobolev = np.mean(rmse_sobolev, axis=1)
    std_sobolev = np.std(rmse_sobolev, axis=1)
    
    # Plot shaded regions for standard deviation
    plt.fill_between(lipschitz_values, 
                     mean_classical - std_classical, 
                     mean_classical + std_classical,
                     alpha=0.3, color='red', label='_nolegend_')
    plt.fill_between(lipschitz_values, 
                     mean_sobolev - std_sobolev, 
                     mean_sobolev + std_sobolev,
                     alpha=0.3, color='blue', label='_nolegend_')
    
    # Plot mean lines
    plt.plot(lipschitz_values, mean_classical, 'r-o', linewidth=2, 
             markersize=6, label='Classical Algorithm', markevery=5)
    plt.plot(lipschitz_values, mean_sobolev, 'b-o', linewidth=2, 
             markersize=6, label='Sobolev Learning', markevery=5)
    
    plt.xlabel('Lipschitz bound', fontsize=12)
    plt.ylabel('Test Error', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.title('Effect of Lipschitz Constant on RMSE', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lipschitz_effect.png', dpi=300, bbox_inches='tight')
    print("  Saved: lipschitz_effect.png")
    plt.close()


def experiment_2_heatmap_analysis():
    """Experiment 2: Effect of Training Data Amount on RMSE and Lipschitz Constant"""
    print("\nRunning Experiment 2: Heatmap Analysis")
    
    # Parameters
    sample_sizes = np.array([10, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    lipschitz_range = np.array([30, 60, 80, 100, 120, 225, 535])
    n_runs_heatmap = 50
    n_test_heatmap = 50
    gamma = 1.0
    lambda_reg = 0.01
    
    # Storage for results
    error_sobolev = np.zeros((len(lipschitz_range), len(sample_sizes)))
    error_classical = np.zeros((len(lipschitz_range), len(sample_sizes)))
    
    for L_idx, L in enumerate(lipschitz_range):
        print(f"  Lipschitz constant: {L}")
        
        for s_idx, n in enumerate(sample_sizes):
            temp_error_sobolev = []
            temp_error_classical = []
            
            for run in range(n_runs_heatmap):
                # Generate data
                X_train, y_train, grad_train = generate_styblinski_tang_data(n, L)
                X_test, y_test, _ = generate_styblinski_tang_data(n_test_heatmap, L)
                
                # Sobolev training
                model_sobolev = SobolevTraining(gamma=gamma, lambda_reg=lambda_reg, beta=1.0)
                model_sobolev.fit(X_train, y_train, grad_train)
                y_pred_sobolev = model_sobolev.predict(X_test)
                temp_error_sobolev.append(np.sqrt(np.mean((y_test - y_pred_sobolev)**2)))
                
                # Classical training
                model_classical = SobolevTraining(gamma=gamma, lambda_reg=lambda_reg, beta=0)
                model_classical.fit(X_train, y_train)
                y_pred_classical = model_classical.predict(X_test)
                temp_error_classical.append(np.sqrt(np.mean((y_test - y_pred_classical)**2)))
            
            error_sobolev[L_idx, s_idx] = np.mean(temp_error_sobolev)
            error_classical[L_idx, s_idx] = np.mean(temp_error_classical)
    
    # Plot Sobolev heatmap
    plt.figure(figsize=(12, 7))
    im = plt.imshow(np.log10(error_sobolev), aspect='auto', cmap='hot', 
                    extent=[sample_sizes[0], sample_sizes[-1], 
                           lipschitz_range[0], lipschitz_range[-1]],
                    origin='lower')
    plt.colorbar(im, label='log10(Test Error)')
    plt.xlabel('No. of samples', fontsize=12)
    plt.ylabel('Lipschitz constant', fontsize=12)
    plt.title('Heatmap for Sobolev Learning Algorithm', fontsize=14)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/heatmap_sobolev.png', dpi=300, bbox_inches='tight')
    print("  Saved: heatmap_sobolev.png")
    plt.close()
    
    # Plot Classical heatmap
    plt.figure(figsize=(12, 7))
    im = plt.imshow(np.log10(error_classical), aspect='auto', cmap='hot',
                    extent=[sample_sizes[0], sample_sizes[-1], 
                           lipschitz_range[0], lipschitz_range[-1]],
                    origin='lower')
    plt.colorbar(im, label='log10(Test Error)')
    plt.xlabel('No. of samples', fontsize=12)
    plt.ylabel('Lipschitz constant', fontsize=12)
    plt.title('Heatmap for Classical Learning Algorithm', fontsize=14)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/heatmap_classical.png', dpi=300, bbox_inches='tight')
    print("  Saved: heatmap_classical.png")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("Sobolev Training in RKHS - Experimental Reproduction")
    print("Based on Zain ul Abdeen et al., IFAC PapersOnLine 56-2 (2023)")
    print("=" * 70)
    
    # Run experiments
    experiment_1_lipschitz_effect()
    experiment_2_heatmap_analysis()
    
    print("\n" + "=" * 70)
    print("Experiments completed!")
    print("Results saved to /mnt/user-data/outputs/")
    print("=" * 70)
