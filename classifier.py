import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
import random

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here
    spam_files, ham_files = file_lists_by_category
    alpha = 1
    vocab = set()
    # Count the number of words in each email
    spam_word_counts = {}
    ham_word_counts = {}
    
    for file in spam_files:
        with open(file ,'r', encoding='latin1') as f:
            words = f.read().split()
            for word in words:  
                spam_word_counts[word] = spam_word_counts.get(word, 0) + 1
                vocab.add(word)
    for file in ham_files:
        with open(file ,'r', encoding='latin1') as f:
            words = f.read().split()
            for word in words:
                ham_word_counts[word] = ham_word_counts.get(word, 0) + 1
                vocab.add(word)
    # Calculate the probabilities
    p_d = {}
    q_d = {}
    print(len(spam_word_counts))
    print(len(ham_word_counts))
    for word in spam_word_counts:
        p_d[word] = (spam_word_counts.get(word, 0) + alpha) / (len(spam_word_counts) + alpha * len(vocab))
    for word in ham_word_counts:
        q_d[word] = (ham_word_counts.get(word, 0) + alpha) / (len(ham_word_counts) + alpha * len(vocab))
    return (p_d, q_d)

def classify_new_email(filename,probabilities_by_category,prior_by_category):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here
    p_d, q_d = probabilities_by_category
    pi, one_minus_pi = prior_by_category
    log_posterior = [0,0]
    with open(filename, 'r', encoding='latin1') as f:
        words = f.read().split()
        for word in words:
            log_posterior[0] += np.log(p_d.get(word, 1e-6))
            log_posterior[1] += np.log(q_d.get(word, 1e-6))
    log_posterior[0] += np.log(pi)
    log_posterior[1] += np.log(one_minus_pi)
    result = ('ham' if log_posterior[0] > log_posterior[1] else 'spam', log_posterior)
    return result

def select_files(directory, fraction=0.7):
    """
    This function builds a customized dataset for each group

    """
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]
    random.shuffle(all_files)
    num_files = int(len(all_files) * fraction)
    return all_files[:num_files]


if __name__ == '__main__':
    
    ############################CHANGE YOUR STUDENT ID###############################
    student1_number = 123456789  # Replace with the actual student number
    student2_number = 123456789  # Replace with the actual student number
    random.seed((student1_number+student2_number)/1000)
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    file_lists = [select_files(folder) for folder in (spam_folder, ham_folder)]
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curveclear


   

 