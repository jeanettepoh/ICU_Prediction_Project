import numpy as np
import pandas as pd
from flask import Flask, request, render_template

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


# Route for a home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            age = int(request.form.get("age")),
            sex = request.form.get("sex"),
            infection = request.form.get("infection"),
            sysbp = int(request.form.get("sysbp")),
            pulse = int(request.form.get("pulse")),
            emergency = request.form.get("emergency")
        )

        input_df = data.convert_input_to_dataframe()
        print(input_df)
        print("Converted input to dataframe")
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(input_df)
        print(results)
        print("Completed model prediction for input")

        return render_template("home.html", results=results)


# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5560)