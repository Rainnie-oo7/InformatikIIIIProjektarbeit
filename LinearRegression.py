import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, \
    recall_score, f1_score
import numpy as np


# 1. Laden der Daten aus einer .csv-Datei
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# 2. Erstellen eines Pipelines für StandardScaler und LinearRegression
def create_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    return pipeline


# 3. Trainieren des Modells
def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)


# 4. Validieren des Modells
def validate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return r2, mae, mse, y_pred


# 5. Hyperparameter-Tuning mit GridSearchCV
def hyperparameter_tuning(X_train, y_train):
    pipeline = create_pipeline()
    param_grid = {
        'regressor__fit_intercept': [True, False],
        'regressor__copy_X': [True, False],
        'regressor__n_jobs': [None, 1, -1],
        'regressor__positive': [True, False]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_


# Klassifizieren der Vorhersagen basierend auf einem Schwellenwert
def classify_predictions(y_pred, threshold):
    return np.where(y_pred >= threshold, 1, 0)


# Hauptprogramm
if __name__ == "__main__":
    # Datei Pfade anpassen
    train_file_path = '''C:/Users/user/path-to-file/breast-cancer_train.csv'''
    test_file_path = '''C:/Users/user/path-to-file/breast-cancer_test.csv'''

    # Laden der Trainingsdaten
    train_data = load_data(train_file_path)

    # Zielspalte anpassen
    target_column = 'diagnosis'

    # Aufteilen der Trainingsdaten
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]

    # Pipeline erstellen
    pipeline = create_pipeline()

    # Modell trainieren
    train_model(pipeline, X_train, y_train)

    # Laden der Testdaten
    test_data = load_data(test_file_path)
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    # Modell validieren
    r2, mae, mse, y_pred = validate_model(pipeline, X_test, y_test)
    print(f'R² on Test Data: {r2}')
    print(f'Mean Absolute Error on Test Data: {mae}')
    print(f'Mean Squared Error on Test Data: {mse}')

    # Hyperparameter-Tuning
    best_params, best_score = hyperparameter_tuning(X_train, y_train)
    print(f'Best Parameters: {best_params}')
    print(f'Best CV Score: {best_score}')

    # Entfernen der Präfixe aus den besten Parametern
    best_params = {key.split('__')[1]: value for key, value in best_params.items()}

    # Erstellen der finalen Pipeline mit den besten Parametern
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression(**best_params))
    ])

    # Trainieren des finalen Modells
    train_model(final_pipeline, X_train, y_train)

    # Validieren des finalen Modells
    final_r2, final_mae, final_mse, final_y_pred = validate_model(final_pipeline, X_test, y_test)
    print(f'Final R² on Test Data: {final_r2}')
    print(f'Final Mean Absolute Error on Test Data: {final_mae}')
    print(f'Final Mean Squared Error on Test Data: {final_mse}')

    # Schwellenwert für die Klassifikation festlegen
    threshold = 0.5  # Dies ist ein Beispielwert. Passen Sie ihn an Ihre Bedürfnisse an.

    # Klassifizieren der Vorhersagen
    y_test_classified = classify_predictions(y_test, threshold)
    y_pred_classified = classify_predictions(final_y_pred, threshold)

    # Berechnen der Klassifikationsmetriken
    accuracy = accuracy_score(y_test_classified, y_pred_classified)
    precision = precision_score(y_test_classified, y_pred_classified)
    recall = recall_score(y_test_classified, y_pred_classified)
    f1 = f1_score(y_test_classified, y_pred_classified)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')

"""
R² on Test Data: 0.6911359869475935
Mean Absolute Error on Test Data: 0.4159563036870681
Mean Squared Error on Test Data: 0.2833816833200384
Best Parameters: {'regressor__copy_X': True, 'regressor__fit_intercept': True, 'regressor__n_jobs': None, 'regressor__positive': False}
Best CV Score: 0.7304826799637941
Final R² on Test Data: 0.6911359869475935
Final Mean Absolute Error on Test Data: 0.4159563036870681
Final Mean Squared Error on Test Data: 0.2833816833200384
Accuracy: 0.8617021276595744
Precision: 1.0
Recall: 0.7851239669421488
F1-Score: 0.8796296296296297
"""
