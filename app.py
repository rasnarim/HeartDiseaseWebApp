

import pickle
import numpy as np
import shap
from flask import Flask, render_template, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Matplotlib
import matplotlib.pyplot as plt

import openai
from markupsafe import Markup
from dotenv import dotenv_values



# OpenAI API Configuration
secrer_keys = dotenv_values(".env")
API_KEY = secrer_keys["API_KEY"]

openai.api_key = API_KEY

# Flask App Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'

# Load the pretrained models
with open('models/logistic_model.pkl', 'rb') as model_file:
    logistic_model = pickle.load(model_file)

with open('models/shap_explainer.pkl', 'rb') as explainer_file:
    shap_explainer = pickle.load(explainer_file)

# Feature names for input
FEATURE_NAMES = [
    "ST_depresssion_exercise",
    "max_HR",
    "number_vessels_involved",
    "age_yr",
    "resting_BP_mm_Hg"
]

# Helper Function: Format OpenAI Reports for Markup
def format_report(report_text):
    report_text = report_text.replace("**", " ")  # Remove unnecessary bold markers
    report_text = report_text.replace("###", "<h2>")
    # report_text = report_text.replace(" - ", "</li><li>")
    report_text = report_text.replace("1. ", "</li></ol><ol><li>")
    return Markup(report_text)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect user inputs for the features
        user_data = [float(request.form[feature]) for feature in FEATURE_NAMES]
        return redirect(url_for('result', data=','.join(map(str, user_data))))

    return render_template('index.html', features=FEATURE_NAMES)

@app.route('/result')
def result():
    # Parse user data from query string
    user_data = list(map(float, request.args.get('data').split(',')))
    user_input = np.array(user_data).reshape(1, -1)

    # Make prediction
    prediction = logistic_model.predict(user_input)[0]
    prediction_proba = logistic_model.predict_proba(user_input)[0, 1]

    # Generate SHAP explanation
    shap_values = shap_explainer(user_input)
    shap.force_plot(
        shap_explainer.expected_value,
        shap_values.values[0],
        user_input[0],
        feature_names=FEATURE_NAMES,
        matplotlib=True
    )

    # Save SHAP force plot
    shap_plot_path = 'static/shap_force_plot.png'
    plt.savefig(shap_plot_path, bbox_inches='tight')
    plt.close()

    # Prepare SHAP values for OpenAI
    shap_results = {
        FEATURE_NAMES[i]: shap_values.values[0][i]
        for i in range(len(FEATURE_NAMES))
    }

    # Format prompt for OpenAI GPT-4
    shap_summary = "\n".join([f"{feature}: {value:.2f}" for feature, value in shap_results.items()])
    openai_prompt = f"""
    A patient was evaluated for heart disease risk using a machine learning model. 
    The model provided the following SHAP analysis for feature importance:

    {shap_summary}

    The prediction probability of heart disease is {prediction_proba:.2%}.

    Please create a structured and well-formatted report in a professional tone suitable for a medical practitioner with the following sections:

    ### 1. Feature Analysis ###
    - Provide a clinical interpretation of each feature, explaining its relevance to heart disease risk.
    - Discuss the patient's specific data and the degree to which each feature influenced the prediction.

    ### 2. Summary of Results ###
    - Concisely summarize the prediction outcomes in a numbered format.
    - Highlight the key features driving the risk assessment and their clinical implications.

    ### 3. Clinical Recommendations ###
    - Provide evidence-based and practical recommendations tailored to the patient's risk factors.
    - Organize recommendations into bullet points, focusing on actionable steps including lifestyle modifications, diagnostic tests, and treatment options.
    - Use clear language emphasizing priorities for patient management and follow-up.

    Ensure the report maintains a professional format, appropriate spacing, and clear headings for easy readability by a medical professional.
    """

    # Call GPT-4 for Report Generation
    try:
        openai_response = openai.ChatCompletion.create(
            model="gpt-4",  # Use GPT-4 for higher-quality responses
            messages=[
                {"role": "system",
                 "content": "You are a medical assistant specializing in patient-friendly, expert-level medical reports. Provide clear explanations in bullet points with structured formatting."},
                {"role": "user", "content": openai_prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        report = openai_response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        report = f"An error occurred while generating the report: {e}"

    # Format the report nicely for display
    formatted_report = f"""
    <h2>Patient Report</h2>
    <h3>Prediction</h3>
    <p><strong>Risk of Heart Disease:</strong> {'High' if prediction else 'Low'} ({prediction_proba:.2%} probability)</p>
    <h3>SHAP Analysis</h3>
    <p>The following features contributed to the prediction:</p>
    <ul>
        {''.join([f"<li><strong>{feature}</strong>: {value:.2f}</li>" for feature, value in shap_results.items()])}
    </ul>
    <h3>Assistant's Explanation</h3>
    <p>{report}</p>
    """


    formatted_report = format_report(report)

    return render_template(
        'result.html',
        prediction=prediction,
        prediction_proba=prediction_proba,
        shap_plot_url=shap_plot_path,
        report=formatted_report
    )

if __name__ == '__main__':
    app.run(debug=True)
