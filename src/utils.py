import os
import sys
import pickle

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logger


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

        logger.info("Object saved to %s", file_path)

    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_models(X_train, y_train, models, params):
    try:
        report = {}

        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        for model_name, model in models.items():
            logger.info("Evaluating model: %s", model_name)

            search = GridSearchCV(
                estimator=model,
                param_grid=params.get(model_name, {}),
                scoring="r2",
                cv=cv,
                n_jobs=-1,
                error_score="raise",
            )

            search.fit(X_train, y_train)

            report[model_name] = {
                "model": search.best_estimator_,
                "cv_score": search.best_score_,
                "best_params": search.best_params_,
            }

            logger.info(
                "%s cross-validation R2: %.4f",
                model_name,
                search.best_score_,
            )

        return report

    except Exception as e:
        raise CustomException(e, sys) from e