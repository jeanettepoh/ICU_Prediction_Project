import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.utils import load_object


@dataclass
class PredictPipelineConfig:
    """
    Provides the parameters for PredictPipeline
    """
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
    model_path = os.path.join("artifacts", "model.pkl")

class PredictPipeline:
    """
    Outputs model prediction when end-user inputs data
    """
    def __init__(self):
        self.pipeline_config = PredictPipelineConfig()

    def predict(self, input):
        """
        Loads and applies saved preprocessor and model objects,
        returns prediction for input data
        """
        try:
            preprocessor_path = self.pipeline_config.preprocessor_path
            model_path = self.pipeline_config.model_path

            preprocessor = load_object(preprocessor_path)
            model, parameters = load_object(model_path)
            model.set_params(**parameters)
            print("Loaded preprocessor and model")

            input_scaled = preprocessor.transform(input)
            preds = model.predict(input_scaled)
            
            outcome_mapping = {0: "Survived", 1: "Died"}
            outcome = outcome_mapping.get(preds[0])
            return outcome
            
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Converts input data into suitable format for further preprocessing and prediction
    """
    def __init__(self,
        age: int,
        sex: str,
        infection: str,
        sysbp: int,
        pulse: int,
        emergency: str
    ):

        self.age = age
        self.sex = sex
        self.infection = infection
        self.sysbp = sysbp
        self.pulse = pulse
        self.emergency = emergency

    def convert_input_to_dataframe(self):
        """
        Maps user input for categorical features to 0 or 1
        before converting entire user input into dataframe
        """
        try:
            categorical_mapping = {
                "Sex": {"Male": 0, "Female": 1},
                "Infection": {"Not Infected": 0, "Infected": 1},
                "Emergency": {"No Emergency": 0, "Emergency": 1}
            }

            self.sex = categorical_mapping["Sex"].get(self.sex)
            self.infection = categorical_mapping["Infection"].get(self.infection)
            self.emergency = categorical_mapping["Emergency"].get(self.emergency)

            custom_data_input_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "infection": [self.infection],
                "sysbp": [self.sysbp],
                "pulse": [self.pulse],
                "emergency": [self.emergency]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)