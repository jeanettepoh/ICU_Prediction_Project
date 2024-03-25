import os
import sys
import yaml
import pickle
import pandas as pd

from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.exception import CustomException
from src.logger import logger


def save_object(file_path, obj):
    """
    Saves given object in pickle format

    Parameters
    ----------
        file_path: Path to save pickle object
        obj: Object to be saved in pickle format

    Returns
    -------
        None 
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads pickle object from given file path and returns it

    Parameters
    ----------
        file_path: Path which contains pickle object
    
    Returns
    -------
        file object in pickle format
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def parse_config(file_path):   
    """
    Parsing function for YAML file

    Parameters
    ----------
        file_path: Path of YAML file

    Return
    ------
        config: Nested Python dictionary of information in YAML file
    """
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
        return config
    
    except Exception as e:
        raise CustomException(e, sys)


def train_and_evaluate(
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        models, 
        params, 
        best_model_file_path
    ):
    """
    This function iterates across all models, performs 
    grid search on training data to find optimum parameters,
    trains model and evaluates performance on testing data

    Parameters
    ----------
        X_train: Training data input
        y_train: Training data outcome
        X_test: Testing data input
        y_test: Testing data outcome
        models: Dictionary of models
        params: Dictionary of model hyperparameters
        best_model_file_path: file path to save best model

    Returns
    -------
        model_performance: Dataframe of evaluation metrics across all models
    """
    try:
        model_performance = pd.DataFrame()

        best_sensitivity = -1
        best_model = None
        best_model_params = None

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            model_params = params[model_name]

            logger.info(f"Performing GridSearchCV for {model_name}")
            grid_search = GridSearchCV(model, model_params, cv=3)

            logger.info(f"Fitting on training data for {model_name}")
            grid_search.fit(X_train, y_train)

            logger.info(f"Making predictions for {model_name}")
            y_pred = grid_search.predict(X_test)
            
            logger.info(f"Generating evaluation metrics for {model_name}")
            performance, sensitivity = generate_evaluation_metrics(y_test, y_pred)
            model_performance[model_name] = performance

            if sensitivity > best_sensitivity:
                best_sensitivity = sensitivity
                best_model_params = grid_search.best_params_
                best_model = grid_search.best_estimator_
       
        best_sensitivity_model = model_performance.loc["Sensitivity"].idxmax()
        logger.info(f"The best sensitivity model is {best_sensitivity_model} with sensitivity {best_sensitivity}")

        logger.info("Saving best sensitivity model with its optimum parameters")
        save_object(
            file_path=best_model_file_path,
            obj=(best_model, best_model_params)
        )

        return model_performance

    except Exception as e:
        raise CustomException(e, sys)


def generate_evaluation_metrics(y_test, y_pred):
    """
    Returns accuracy, f1 score, precision, sensitivity, specificity
    and roc auc score between y_test and y_preds

    Parameters
    ----------
        y_test: List of Actual values (0 or 1)
        y_pred: List of Predicted values (0 or 1)

    Returns
    -------
        performance: pd series of all evalution metrics for given model
        sensitivity: Sensitivity score for given model to identify best sensitivity model
    """
    try:
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        sensitivity = sensitivity_score(y_test, y_pred)
        specificity = specificity_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        
        performance = pd.Series({'Accuracy': accuracy, 'F1': f1, 'Precision': precision, 'Sensitivity': sensitivity,
                                'Specificity': specificity, 'ROC AUC': roc_auc})

        return performance, sensitivity

    except Exception as e:
        raise CustomException(e, sys)