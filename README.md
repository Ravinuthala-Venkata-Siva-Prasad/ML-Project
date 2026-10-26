# Student Exam Performance Prediction

An end-to-end Machine Learning project that predicts students' mathematics scores based on academic and demographic information.

## Project Overview

This project demonstrates a complete machine learning workflow, from data preprocessing and model training to prediction through a Flask web application.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost and CatBoost
- Flask
- HTML and CSS

## Features

- Data ingestion and validation
- Data preprocessing using Scikit-learn pipelines
- Multiple regression algorithms
- Hyperparameter tuning
- Model evaluation using R² score
- Web interface for predicting mathematics scores
- Modular Python project structure

## Installation

```bash
git clone https://github.com/Ravinuthala-Venkata-Siva-Prasad/Student-Performance-Prediction.git
cd Student-Performance-Prediction
pip install -r requirements.txt
```

## Train the Model

```bash
python -m src.pipeline.train_pipeline
```

## Run the Application

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

See the LICENSE file for details.
