"""
Sobolev Training in RKHS - Optimized Experimental Reproduction
Based on "A Theoretical Analysis of Using Gradient Data for Sobolev Training in RKHS"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(42)

class SobolevTraining:
    def __init__(self, gamma=1.0, lambda_reg=0.01, beta=1.0):
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.beta = beta
        self.alpha = None
        self.X_train = None
        
    def gaussian_kernel(self, X1, X2):
        dist_sq = cdist(X1, X2, metric='sqeuclidean')
        return np.exp(-dist_sq / (2 * self.gamma**2))
    
    def gradient_kernel(self, X1, X2, dim=0):
        K = self.gaussian_kernel(X1, X2)
        diff = X1[:, dim:dim+1] - X2[:, dim:dim+1].T
        return -(diff / self.gamma**2) * K
    
    def fit(self, X_train, y_train, grad_train=None):
        self.X_train = np.array(X_train)
        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)
        
        y_train = np.array(y_train).flatten()
        n, d = self.X_train.shape
        
        K = self.gaussian_kernel(self.X_train, self.X_train)
        
        if self.beta == 0 or grad_train is None:
            self.alpha = np.linalg.solve(K + self.lambda_reg * np.eye(n), y_train)
        else:
            grad_train = np.array(grad_train)
            if grad_train.ndim == 1:
                grad_train = grad_train.reshape(-1, 1)
            
            K_aug = K.copy()
            y_aug = y_train.copy()
            
            for j in range(d):
                K_grad_j = self.gradient_kernel(self.X_train, self.X_train, j)
                K_aug += self.beta * K_grad_j.T @ K_grad_j
                y_aug += self.beta * K_grad_j.T @ grad_train[:, j]
            
            self.alpha = np.linalg.solve(K_aug + self.lambda_reg * np.eye(n), y_aug)
        
        return self
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        K_test = self.gaussian_kernel(X_test, self.X_train)
        return K_test @ self.alpha


def generate_data(n, L):
    X = 2 * np.random.rand(n, 1) - 1
    scale = L / 50.0
    y = scale * (X**4 - 16*X**2 + 5*X).flatten() / 2
    grad_y = scale * (4*X**3 - 32*X + 5) / 2
    y += 0.01 * np.random.randn(n)
    grad_y += 0.01 * np.random.randn(n, 1)
    return X, y, grad_y


print("Running Experiment 1: Lipschitz Effect (Reduced)")

# Reduced parameters for faster execution
n_train = 30
n_test = 20
n_runs = 20  # Reduced from 100
lipschitz_values = np.arange(20, 510, 20)  # Coarser sampling

rmse_classical = []
rmse_sobolev = []

for L in lipschitz_values:
    temp_classical = []
    temp_sobolev = []
    
    for _ in range(n_runs):
        X_train, y_train, grad_train = generate_data(n_train, L)
        X_test, y_test, _ = generate_data(n_test, L)
        
        # Classical
        model = SobolevTraining(gamma=1.0, lambda_reg=0.01, beta=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        temp_classical.append(np.sqrt(np.mean((y_test - y_pred)**2)))
        
        # Sobolev
        model = SobolevTraining(gamma=1.0, lambda_reg=0.01, beta=1.0)
        model.fit(X_train, y_train, grad_train)
        y_pred = model.predict(X_test)
        temp_sobolev.append(np.sqrt(np.mean((y_test - y_pred)**2)))
    
    rmse_classical.append(temp_classical)
    rmse_sobolev.append(temp_sobolev)
    print(f"  Processed L={L}")

rmse_classical = np.array(rmse_classical)
rmse_sobolev = np.array(rmse_sobolev)

# Plot
plt.figure(figsize=(10, 7))
mean_c = np.mean(rmse_classical, axis=1)
std_c = np.std(rmse_classical, axis=1)
mean_s = np.mean(rmse_sobolev, axis=1)
std_s = np.std(rmse_sobolev, axis=1)

plt.fill_between(lipschitz_values, mean_c - std_c, mean_c + std_c,
                 alpha=0.3, color='red')
plt.fill_between(lipschitz_values, mean_s - std_s, mean_s + std_s,
                 alpha=0.3, color='blue')
plt.plot(lipschitz_values, mean_c, 'r-o', linewidth=2, label='Classical Algorithm')
plt.plot(lipschitz_values, mean_s, 'b-o', linewidth=2, label='Sobolev Learning')

plt.xlabel('Lipschitz bound', fontsize=12)
plt.ylabel('Test Error', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.title('Effect of Lipschitz Constant on RMSE', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/lipschitz_effect.png', dpi=300, bbox_inches='tight')
print("Saved: lipschitz_effect.png")

# Experiment 2: Heatmap
print("\nRunning Experiment 2: Heatmap (Reduced)")

sample_sizes = [10, 50, 100, 200, 500, 1000, 2000]  # Reduced
lipschitz_range = [30, 60, 100, 150, 300, 500]  # Reduced
n_runs = 10  # Reduced

error_sobolev = np.zeros((len(lipschitz_range), len(sample_sizes)))
error_classical = np.zeros((len(lipschitz_range), len(sample_sizes)))

for i, L in enumerate(lipschitz_range):
    for j, n in enumerate(sample_sizes):
        temp_s = []
        temp_c = []
        
        for _ in range(n_runs):
            X_train, y_train, grad_train = generate_data(n, L)
            X_test, y_test, _ = generate_data(20, L)
            
            model = SobolevTraining(gamma=1.0, lambda_reg=0.01, beta=1.0)
            model.fit(X_train, y_train, grad_train)
            temp_s.append(np.sqrt(np.mean((y_test - model.predict(X_test))**2)))
            
            model = SobolevTraining(gamma=1.0, lambda_reg=0.01, beta=0)
            model.fit(X_train, y_train)
            temp_c.append(np.sqrt(np.mean((y_test - model.predict(X_test))**2)))
        
        error_sobolev[i, j] = np.mean(temp_s)
        error_classical[i, j] = np.mean(temp_c)
    print(f"  Processed L={L}")

# Sobolev heatmap
plt.figure(figsize=(12, 7))
im = plt.imshow(np.log10(error_sobolev), aspect='auto', cmap='hot',
                extent=[sample_sizes[0], sample_sizes[-1], 
                       lipschitz_range[0], lipschitz_range[-1]],
                origin='lower', interpolation='bilinear')
plt.colorbar(im, label='log10(Test Error)')
plt.xlabel('No. of samples', fontsize=12)
plt.ylabel('Lipschitz constant', fontsize=12)
plt.title('Heatmap for Sobolev Learning Algorithm', fontsize=14)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/heatmap_sobolev.png', dpi=300, bbox_inches='tight')
print("Saved: heatmap_sobolev.png")

# Classical heatmap
plt.figure(figsize=(12, 7))
im = plt.imshow(np.log10(error_classical), aspect='auto', cmap='hot',
                extent=[sample_sizes[0], sample_sizes[-1], 
                       lipschitz_range[0], lipschitz_range[-1]],
                origin='lower', interpolation='bilinear')
plt.colorbar(im, label='log10(Test Error)')
plt.xlabel('No. of samples', fontsize=12)
plt.ylabel('Lipschitz constant', fontsize=12)
plt.title('Heatmap for Classical Learning Algorithm', fontsize=14)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/heatmap_classical.png', dpi=300, bbox_inches='tight')
print("Saved: heatmap_classical.png")

print("\nExperiments completed!")
