import sys
from pathlib import Path
import pandas as pd
import os

from src.exception import CustomException
from src.utils import load_object

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            model = load_object(str(model_path))
            preprocessor = load_object(str(preprocessor_path))

            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)

            return predictions

        except Exception as e:
            raise CustomException(e, sys) from e