from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Assuming you have your dataset and labels prepared as X and y

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grids to search
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],  # Laplace smoothing parameter (alpha)
    'class_prior': [None, [0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]  # Positive class prior
}

# Create a Multinomial Naive Bayes classifier
naive_bayes = MultinomialNB()

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=naive_bayes, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the classifier with the best parameters on the entire training set
best_classifier = MultinomialNB(alpha=best_params['alpha'], class_prior=best_params['class_prior'])
best_classifier.fit(X_train, y_train)

# Evaluate the best classifier on the test set
accuracy = best_classifier.score(X_test, y_test)

print("Best Laplace smoothing parameter (alpha):", best_params['alpha'])
print("Best Positive class prior:", best_params['class_prior'])
print("Accuracy on test set:", accuracy)
