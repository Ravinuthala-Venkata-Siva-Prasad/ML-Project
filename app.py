
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


# Home page
@app.route("/")
def index():
    return render_template("index.html")


# Prediction page
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "GET":
        return render_template("home.html")

    try:
        # Get student details from HTML form
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get(
                "test_preparation_course"
            ),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score"))
        )

        # Convert input data to DataFrame
        pred_df = data.get_data_as_data_frame()

        print("Input Data:")
        print(pred_df)

        # Load trained model and predict
        predict_pipeline = PredictPipeline()

        print("Before Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        # Display prediction
        return render_template(
            "home.html",
            results=round(float(results[0]), 2)
        )

    except Exception as e:
        app.logger.exception("Prediction failed")

        return render_template(
            "home.html",
            error="Prediction failed. Check the terminal for details."
        ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

