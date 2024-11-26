import pickle
import numpy as np
import shap
from flask import Flask, render_template, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Matplotlib
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from markupsafe import Markup
from dotenv import dotenv_values

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

# Hugging Face BioMistral-7B Configuration
model_name = "BioMistral/BioMistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load model without GPU-specific optimizations
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             low_cpu_mem_usage=True,
                                             offload_folder="./offload")
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Helper Function: Format BioMistral Reports for Markup
def format_report(report_text):
    report_text = report_text.replace("**", " ")  # Remove unnecessary bold markers
    report_text = report_text.replace("###", "<h2>")
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

    # Prepare SHAP values for BioMistral
    shap_results = {
        FEATURE_NAMES[i]: shap_values.values[0][i]
        for i in range(len(FEATURE_NAMES))
    }

    # Format prompt for BioMistral-7B
    shap_summary = "\n".join([f"{feature}: {value:.2f}" for feature, value in shap_results.items()])
    prompt = f"""
    A patient was evaluated for heart disease risk using a machine learning model. 
    The model provided the following SHAP analysis for feature importance:

    {shap_summary}

    The prediction probability of heart disease is {prediction_proba:.2%}.

    Create a detailed, professional report in a structured format suitable for a cardiologist. Include the following sections:

    ### 1. Feature Analysis ###
    - For each feature, explain its clinical relevance in the context of heart disease.
    - Interpret the patient's specific data and explain how each feature contributes to the prediction.

    ### 2. Summary of Results ###
    - Summarize the key takeaways in a numbered format.
    - Include insights into the patient's risk profile and highlight the critical factors influencing the outcome.

    ### 3. Clinical Recommendations ###
    - Provide evidence-based recommendations tailored to the patient's risk factors.
    - Break recommendations into:
        - Lifestyle modifications
        - Diagnostic tests
        - Treatment options
    - Use clear, action-oriented language suitable for medical follow-up.

    ### 4. Supporting Insights ###
    - Offer additional clinical insights or research references to contextualize the patient's risk assessment.
    - Suggest potential next steps or areas for further evaluation.
    """


    # Call BioMistral-7B for Report Generation
    try:
        response = text_generator(prompt, max_length=1200, num_return_sequences=1, temperature=0.6, top_p=0.9)
        report = response[0]["generated_text"]
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
