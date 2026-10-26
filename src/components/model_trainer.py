import sys
from pathlib import Path
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = str(
        PROJECT_ROOT / "artifacts" / "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
    ):
        try:
            models = {
                "Random Forest": RandomForestRegressor(
                    random_state=42
                ),
                "Decision Tree": DecisionTreeRegressor(
                    random_state=42
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    random_state=42
                ),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror",
                    n_jobs=1,
                ),
                "CatBoost": CatBoostRegressor(
                    verbose=0,
                    random_seed=42,
                    allow_writing_files=False,
                ),
                "AdaBoost": AdaBoostRegressor(
                    random_state=42
                ),
            }

            params = {
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10],
                },
                "Decision Tree": {
                    "max_depth": [None, 5, 10],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                },
                "Linear Regression": {},
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5],
                },
                "CatBoost": {
                    "iterations": [100, 200],
                    "depth": [4, 6],
                    "learning_rate": [0.03, 0.1],
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                },
            }

            report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                models=models,
                params=params,
            )

            best_model_name = max(
                report,
                key=lambda name: report[name]["cv_score"],
            )

            best_model = report[best_model_name]["model"]

            test_predictions = best_model.predict(X_test)
            test_r2 = r2_score(y_test, test_predictions)

            save_object(
                self.model_trainer_config.trained_model_file_path,
                best_model,
            )

            logger.info("Best model: %s", best_model_name)
            logger.info(
                "Best cross-validation R2: %.4f",
                report[best_model_name]["cv_score"],
            )
            logger.info("Test R2: %.4f", test_r2)

            return {
                "best_model_name": best_model_name,
                "cross_validation_r2": report[best_model_name]["cv_score"],
                "test_r2": test_r2,
                "best_parameters": report[best_model_name]["best_params"],
            }

        except Exception as e:
            logger.exception("Model training failed")
            raise CustomException(e, sys) from e