from flask import Flask, request, render_template_string
import pickle
import pandas as pd

# Load trained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# HTML template with previous inputs preserved and Clear button
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Maternal Risk Prediction</title>
</head>
<body>
    <h2>Maternal Risk Level Prediction</h2>
    <form method="post">
        Age: <input type="number" name="age" step="1" required value="{{ age }}"><br>
        BMI: <input type="number" name="bmi" step="0.1" required value="{{ bmi }}"><br>
        Systolic BP: <input type="number" name="systolic_bp" step="1" required value="{{ systolic_bp }}"><br>
        Diastolic BP: <input type="number" name="diastolic" step="1" required value="{{ diastolic }}"><br>
        Blood Sugar: <input type="number" name="bs" step="0.1" required value="{{ bs }}"><br>
        Previous Complications (0/1): <input type="number" name="previous_complications" step="1" min="0" max="1" required value="{{ previous_complications }}"><br>
        Preexisting Diabetes (0/1): <input type="number" name="preexisting_diabetes" step="1" min="0" max="1" required value="{{ preexisting_diabetes }}"><br>
        Gestational Diabetes (0/1): <input type="number" name="gestational_diabetes" step="1" min="0" max="1" required value="{{ gestational_diabetes }}"><br>
        Mental Health (0/1): <input type="number" name="mental_health" step="1" min="0" max="1" required value="{{ mental_health }}"><br>
        Heart Rate: <input type="number" name="heart_rate" step="1" required value="{{ heart_rate }}"><br><br>
        <input type="submit" value="Predict">
        <input type="submit" name="clear" value="Clear">
    </form>

    {% if prediction is defined %}
        <h3>Prediction Result:</h3>
        <p>Predicted Risk Level: <b>{{ prediction }}</b></p>
        <p>Probability of High Risk: <b>{{ probability }}</b></p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def predict():
    # Default values for the form fields
    form_values = {
        'age': '',
        'bmi': '',
        'systolic_bp': '',
        'diastolic': '',
        'bs': '',
        'previous_complications': '',
        'preexisting_diabetes': '',
        'gestational_diabetes': '',
        'mental_health': '',
        'heart_rate': ''
    }

    prediction = None
    probability = None

    if request.method == "POST":
        # Clear button pressed
        if 'clear' in request.form:
            return render_template_string(html_template, **form_values)

        # Collect form data
        for key in form_values.keys():
            form_values[key] = request.form[key]

        input_data = {k: [float(v) if k not in ['previous_complications', 'preexisting_diabetes', 'gestational_diabetes', 'mental_health'] else int(v)]
                      for k, v in form_values.items()}

        column_order = ['age', 'systolic_bp', 'diastolic', 'bs', 'bmi',
                        'previous_complications', 'preexisting_diabetes',
                        'gestational_diabetes', 'mental_health', 'heart_rate']

        df = pd.DataFrame(input_data)[column_order]

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        prediction = "High Risk" if pred == 1 else "Low Risk"
        probability = round(prob, 3)

    return render_template_string(html_template, prediction=prediction, probability=probability, **form_values)

if __name__ == "__main__":
    app.run(debug=True)
