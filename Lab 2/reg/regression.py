import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import util

# the customed a1 and a0 are:  -0.5 0.6

inv = np.linalg.inv
var = np.var
det = np.linalg.det
pi = np.pi
log = np.log
sqrt = np.sqrt
exp = np.exp 
sum = np.sum
mean = np.mean
dot = np.dot

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    fx = lambda x: exp(-x**2 / (2*beta**2)) / sqrt(2*pi*beta**2)
    x_i = np.linspace(-1, 1, 1000)
    y_i = np.linspace(-1, 1, 1000)
    X, Y = np.meshgrid(x_i, y_i)
    Z = fx(X) * fx(Y)
    plt.title('Prior Distribution')
    plt.contour(X, Y, Z)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    """

    x = x.reshape(-1, 1) if x.ndim == 1 else x
    X = np.column_stack([np.ones(len(x)), x])
    
    # Calculate posterior mean and covariance
    mu_a_xz = inv(X.T @ X + (sigma2/beta**2) * np.eye(2)) @ X.T @ z
    mu_a_xz = mu_a_xz.flatten()  # Convert to 1D array
    cov_a_xz = sigma2 * inv(X.T @ X + (sigma2/beta**2) * np.eye(2))
    
    # Create grid for contour plot
    a0 = np.linspace(mu_a_xz[0] - 2*sqrt(cov_a_xz[0,0]), 
                          mu_a_xz[0] + 2*sqrt(cov_a_xz[0,0]), 100)
    a1 = np.linspace(mu_a_xz[1] - 2*sqrt(cov_a_xz[1,1]), 
                          mu_a_xz[1] + 2*sqrt(cov_a_xz[1,1]), 100)
    A0, A1 = np.meshgrid(a0, a1)
    pos = np.dstack((A0, A1))
    rv = stats.multivariate_normal(mean=mu_a_xz, cov=cov_a_xz)
    Z = rv.pdf(pos)

    #Display the posterior distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contour(A0, A1, Z, levels=20)
    plt.plot(mu_a_xz[0], mu_a_xz[1], 'r*', label='MAP estimate')
    plt.xlabel('a0')
    plt.ylabel('a1')
    plt.title(f'Posterior Distribution with {len(x)} samples')
    plt.legend()
    
    # Data fit subplot
    plt.subplot(1, 2, 2)
    x_plot = np.linspace(-4, 4, 100)
    y_plot = mu_a_xz[0] + mu_a_xz[1] * x_plot
    plt.scatter(x, z, color='blue', label='Training data')
    plt.plot(x_plot, y_plot, 'r-', label='MAP fit')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Data and MAP Fit')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'posterior{len(x)}.pdf')
    plt.show()
    
    return (mu_a_xz, cov_a_xz)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    """
    x_pred = np.array(x).reshape(-1, 1)
    X_pred = np.column_stack([np.ones_like(x_pred), x_pred])  # Add bias term
    
    # Get mean and variance of the posterior
    y_mean = X_pred @ mu
    y_var = np.diag(X_pred @ Cov @ X_pred.T) + sigma2
    y_std = np.sqrt(y_var)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train, z_train, color='blue', label='Training data')
    plt.errorbar(x_pred.flatten(), y_mean.flatten(), yerr=y_std, 
                 fmt='ro', capsize=5, label='Predictions with std dev')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title(f'Prediction Distribution with {len(x_train)} samples')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'predict{len(x_train)}.pdf')
    plt.show()
    
    return

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = [1,5,100]

    # prior distribution p(a)
    priorDistribution(beta)
    
    for n in ns:
        # used samples
        x = x_train[0:n]
        z = z_train[0:n]
        mu, cov = posteriorDistribution(x,z,beta,sigma2)
        predictionDistribution(x_test,beta,sigma2,mu,cov,x,z)
        

    

    
    
    

    
