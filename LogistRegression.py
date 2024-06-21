import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

train_file_path = '''C:/Users/Boris Grillborzer/Nextcloud/4SEMESTERInformatik/UB6Projekt/breast-cancer_train.csv'''
test_file_path = '''C:/Users/Boris Grillborzer/Nextcloud/4SEMESTERInformatik/UB6Projekt/breast-cancer_test.csv'''

# Load your training data
df_train = pd.read_csv(train_file_path)

# Split the data into features (X) and labels (y)
X_train = df_train.drop('diagnosis', axis=1)
y_train = df_train['diagnosis']

# Split the training data into training and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Scale your data to avoid limit-reach iter Warning from optimisation Algo L-BFGS. Hier: Liblinear, StandardScaler. Es gibt: MinMaxScaler, newton-cg, oder cross-validation to select the best solver.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
# Train a logistic regression model on the training set
logreg = LogisticRegression(solver='liblinear', max_iter=1000)
logreg.fit(X_train_split, y_train_split)


# Evaluate the model on the validation set
y_pred_val = logreg.predict(X_val)
accuracy_val = accuracy_score(y_val, y_pred_val)
print(f'Validation accuracy: {accuracy_val:.3f}')
# __________________________________________________________________________
# Use the test data to get an unbiased estimate of the model's performance
# ... (load test data and evaluate the model)
df_test = pd.read_csv(test_file_path)
X_test = df_test.drop('diagnosis', axis=1)
y_test = df_test['diagnosis']

X_test_scaled = scaler.transform(X_test)
# Evaluate the model on the test data
y_pred_test = logreg.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)
f1_test = f1_score(y_test, y_pred_test)

print(f'Test accuracy: {accuracy_test:.3f}')
print(f'Test precision: {precision_test:.3f}')
print(f'Test recall: {recall_test:.3f}')
print(f'Test F1-score: {f1_test:.3f}')
# __________________________________________________________________________

# Define the hyperparameters to tune
param_grid = {
    'penalty': ['l1', 'l2'],  # L1 (Lasso) or L2 (Ridge) regularization
    'C': [0.1, 1.0, 10.0],  # inverse of regularization strength
    'max_iter': [50, 100, 300]  # maximum number of iterations
}

# Create a LogisticRegression instance with the default parameters
lr_model = LogisticRegression()

# Create a GridSearchCV object with the hyperparameter grid and the evaluation metric
grid_search = GridSearchCV(lr_model, param_grid, scoring='f1_macro')

# Fit the GridSearchCV object to your training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use the best hyperparameters to train a new LogisticRegression instance
best_lr_model = LogisticRegression(**grid_search.best_params_)
best_lr_model.fit(X_train, y_train)

# Evaluate the best model on your test data
y_pred = best_lr_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("F1-score:", f1_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))

"""                  ##################### ohne Hypa-Tu
Validation accuracy: 0.935
Test accuracy: 0.963
Test precision: 0.967
Test recall: 0.975
Test F1-score: 0.971
"""

""" # 100, 500, 1000  #################### 1
Best parameters: {'C': 10.0, 'max_iter': 500, 'penalty': 'l2'}
Best score: 0.9552672431695572
"""

"""
Accuracy: 0.9574468085106383
Precision: 0.9511630739252223
F1-score: 0.9539215686274509
Recall: 0.9569507832737141
"""

""" # 50, 100, 300   ##################### 2
Best parameters: {'C': 10.0, 'max_iter': 300, 'penalty': 'l2'}
Best score: 0.9551117384199891
"""
"""
Accuracy: 0.9680851063829787
Precision: 0.9626111314090853
F1-score: 0.9654411764705881
Recall: 0.9685457012458369
"""

# Die Accuracy, Precision, F1-Score und Recall-Test Werte sind schlechter geworden! Wie kann das sein?

# The age-old problem of hyperparameter tuning woes!
#
# When you see a decrease in performance metrics (e.g., accuracy, precision, F1-score, recall) after hyperparameter tuning,
# it's often due to overfitting or over-regularization. Here are some possible reasons:
#
#     Overfitting: The model becomes too specialized to the training data and fails to generalize well to new, unseen data
#     (i.e., the test data). This can happen when the model is too complex or when the regularization strength is too high.
#     Over-regularization: The model is regularized too much, which can lead to underfitting. This occurs when the regularization
#     strength is too high, causing the model to become too simple and lose its ability to capture important patterns in the data.
#
# To address these issues, you can try the following:
#
#     Increase the regularization strength: If you're using L1 (Lasso) or L2 (Ridge) regularization, try increasing the
#       regularization strength (C parameter) to reduce overfitting.
#     Decrease the regularization strength: If you're using L1 or L2 regularization, try decreasing the regularization strength
#       (C parameter) to reduce underfitting.
#     Use a different regularization technique: Consider using a different regularization method, such as dropout or
#       early stopping, which can help prevent overfitting.
#     Increase the model's complexity: If your model is too simple, try adding more features, increasing the number of
#       hidden layers in a neural network, or using a more complex model architecture.
#     Regularly monitor and adjust hyperparameters: Perform grid search or random search on a validation set to find the
#       optimal hyperparameters that balance overfitting and underfitting.
#     Use techniques like early stopping: Implement early stopping to stop training when the model's performance on the
#       validation set starts to degrade.
#     Try different evaluation metrics: If you're using a specific metric like F1-score, try switching to another metric
#       like accuracy or precision-recall curve to get a better understanding of your model's performance.
#
# Remember that hyperparameter tuning is an iterative process. Analyze your results, adjust your hyperparameters,
# and re-train your model until you achieve the desired level of performance.