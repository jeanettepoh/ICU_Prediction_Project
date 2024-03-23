import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object, parse_config, train_and_evaluate


@dataclass
class ModelTrainerConfig:
    """
    Provides the parameters for ModelTrainer
    """
    hyper_params_file_path = os.path.join("config", "params.yaml")
    best_model_file_path = os.path.join("artifacts", "model.pkl")
    saved_results_file_path = os.path.join("results", "model_performance.csv")


class ModelTrainer:
    """
    Performs hyperparameter search and training process for given 
    models, then evaluates model performance on testing data
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        """
        Initiates training and saves summary of model performance in csv format
        """
        try:
            os.makedirs(os.path.dirname(self.model_trainer_config.saved_results_file_path), exist_ok=True)

            models = {
                "Logistic Regression": LogisticRegression(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42)
            }

            logger.info("Loading hyperparameters for grid search and evaluation")
            params = parse_config(self.model_trainer_config.hyper_params_file_path)

            model_performance = train_and_evaluate(
                    X_train = X_train,
                    y_train = y_train,
                    X_test = X_test,
                    y_test = y_test,
                    models = models,
                    params = params,
                    best_model_file_path = self.model_trainer_config.best_model_file_path                 
                )

            model_performance.to_csv(self.model_trainer_config.saved_results_file_path, index=False, header=True)
            logger.info("Saved model performance in csv format")


        except Exception as e:
            raise CustomException(e, sys)


# Check if above code works
if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    from src.components.data_transformation import DataTransformation
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test = \
            data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)