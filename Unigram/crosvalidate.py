from sklearn.model_selection import KFold
import numpy as np
import naive_bayes as nb
import reader as rd
import mp1

def cross_validate_with_dev(train_set, train_labels, dev_set, dev_labels, laplace_values, pos_prior_values):
    # Initialize the results array
    results = np.zeros((len(laplace_values), len(pos_prior_values)))
    
    # Iterate over the different values of the hyperparameters
    for i, laplace in enumerate(laplace_values):
        for j, pos_prior in enumerate(pos_prior_values):
            # Train the model on the training set
            predicted_labels = nb.naiveBayes(dev_set, train_set, train_labels, laplace, pos_prior)
            
            # Evaluate the model on the development set
            accuracy, _, _, _, _ = mp1.compute_accuracies(predicted_labels, dev_labels)
            
            # Update the results array
            results[i, j] = accuracy
    
    # Find the best values for the hyperparameters
    best_index = np.unravel_index(np.argmax(results), results.shape)
    best_laplace = laplace_values[best_index[0]]
    best_pos_prior = pos_prior_values[best_index[1]]
    
    # Print the best hyperparameters
    print(f"Best laplace: {best_laplace}")
    print(f"Best pos_prior: {best_pos_prior}")
    
    return best_laplace, best_pos_prior

# Example usage
laplace_values = [0.1, 0.5, 1.0, 1.5, 2.0]
pos_prior_values = [0.1, 0.3, 0.5, 0.7, 0.9]

train_set, train_labels, dev_set, dev_labels = nb.load_data(r'F:\NLP\data\movie_reviews\train',r'F:\NLP\data\movie_reviews\dev',stemming=False, lowercase=False,)

best_laplace, best_pos_prior = cross_validate_with_dev(train_set, train_labels, dev_set, dev_labels, laplace_values, pos_prior_values)

