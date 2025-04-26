import numpy as np
import matplotlib.pyplot as plt
import util

pi_1 = 0.5
pi_2 = 0.5

inv = np.linalg.inv
det = np.linalg.det
pi = np.pi
log = np.log
sqrt = np.sqrt
exp = np.exp 
sum = np.sum
mean = np.mean

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the 2D features of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_1,mu_2,cov,cov_1,cov_2
    in which mu_1, mu_2 are mean vectors (as 1D arrays)
             cov, cov_1, cov_2 are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here

    # Meshgrid and Indices
    x_i = np.linspace(-4, 6, 1000)
    y_i = np.linspace(-5, 5, 1000)
    mesh_x, mesh_y = np.meshgrid(x_i, y_i)

    # Retrieve Mean, Covariance
    n_male, n_female = sum(y == 1), sum(y==2)
    mu_male, mu_female = mean(x[y==1], axis=0), mean(x[y==2], axis=0)
    cov_male = ((x[y==1]-mu_male).T@(x[y==1]-mu_male))/n_male
    cov_female = ((x[y==2]-mu_female).T@(x[y==2]-mu_female))/n_female
    cov = (n_male*cov_male + n_female*cov_female) / len(x)
    
    # Plotting the Conditional Gaussian distributions
    coord = np.dstack((mesh_x, mesh_y))
    print("Coord shape: ",coord.shape)
    print("Mu male shape: ",mu_male.shape)
    print("Cov male shape: ",cov_male.shape)
    diff_male = coord - mu_male
    fxy_male_plot = (1/(2*pi*sqrt(det(cov_male)))) * \
                    exp(-0.5 * sum(diff_male @ inv(cov_male) * diff_male, axis=2))
    fxy_male = plt.contour(mesh_x, mesh_y, fxy_male_plot, colors='r')
    diff_female = coord - mu_female
    fxy_female_plot = (1/(2*pi*sqrt(det(cov_female)))) * exp(-0.5 * sum(diff_female @ inv(cov_female) * diff_female, axis=2))
    fxy_female = plt.contour(mesh_x, mesh_y, fxy_female_plot, colors='b')
    
    # Linear Gaussian Discriminant:
    beta_1 = mu_male.T@inv(cov)
    beta_2 = mu_female.T@inv(cov)
    gamma_1 = log(pi_1) - 0.5*mu_male.T@inv(cov)@mu_male
    gamma_2 = log(pi_2) - 0.5*mu_female.T@inv(cov)@mu_female
    lgd = lambda x,y: (1/(beta_2[1]-beta_1[1])) * ((beta_1[0]-beta_2[0]) * x + (gamma_1 - gamma_2)) - y
    # lgd_contour = plt.contour(x_i, y_i, lgd(mesh_x, mesh_y), colors='r', levels=[0])
    # lgd_h,_ = lgd_contour.legend_elements()

    # Quadratic Gaussian Discriminant:
    a_1 = inv(cov_male)[0,0] - inv(cov_female)[0,0]
    a_2 = inv(cov_male)[0,1] + inv(cov_male)[1,0] - inv(cov_female)[0,1] - inv(cov_female)[1,0]
    a_3 = inv(cov_male)[1,1] - inv(cov_female)[1,1]
    b_1 = mu_male.T@(inv(cov_male))
    b_2 = mu_female.T@(inv(cov_female))
    c_1 = log(pi_1) - 0.5*log(det(cov_male)) - 0.5*mu_male.T@inv(cov_male)@mu_male
    c_2 = log(pi_2) - 0.5*log(det(cov_female)) - 0.5*mu_female.T@inv(cov_female)@mu_female
    qgd = lambda x, y: -0.5*a_1*x**2 -0.5*a_2*x*y -0.5*a_3*y**2 + (b_1[0]-b_2[0])*x + (b_1[1]-b_2[1])*y + (c_1 - c_2)
    qgd_contour = plt.contour(x_i, y_i, qgd(mesh_x, mesh_y), colors='g', levels=[0])
    qgd_h,_ = qgd_contour.legend_elements()

    # Training data:
    males = plt.plot(x[y==1][:,0], x[y==1][:,1], 'ro')
    females  = plt.plot(x[y==2][:,0], x[y==2][:,1], 'bo')
    plt.ylim(-5, 5)
    plt.legend([males[0], females[0], qgd_h[0]], ['Male', 'Female', 'QGD Decision Boundary'])
    # plt.legend([males[0], females[0], lgd_h[0], qgd_h[0]], ['Male', 'Female', 'LGD Decision boundary', 'QDA Decision Boundary'])
    plt.show()  
    return (mu_male,mu_female,cov,cov_male,cov_female,lgd,qgd)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y,lgd,qgd):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_1,mu_2,cov,cov_1,mu_2: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the 2D features of the N samples 
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
    mis_lda,mis_qda = 0,0
    y = (y-1.5)/0.5

    for x_i,y_i in zip(x,y):
        if np.sign(lgd(x_i[0],x_i[1])) != np.sign(y_i):
            mis_lda += 1
        if np.sign(qgd(x_i[0],x_i[1])) == np.sign(y_i):
            mis_qda += 1
    mis_lda = mis_lda / len(x)
    mis_qda = mis_qda / len(x)
    
    print(f"Misclassfication Rate for Linear Discriminant Analysis: {mis_lda*100:2f}%")
    print(f"Misclassfication Rate for Quadratic Discriminant Analysis: {mis_qda*100:2f}%")
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainData.txt')
    x_test, y_test = util.get_data_in_file('testData.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female,lgd,qgd = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test,lgd,qgd)
    

    
    
    

    
