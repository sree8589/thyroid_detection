from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import json

app = Flask(__name__)

# Load model and encoders
model = joblib.load(r"D:\anaco\caramia\xgb_best_model.pkl")
label_encoders = joblib.load(r"D:\anaco\caramia\label_encoders.pkl")
y_encoder = joblib.load(r"D:\anaco\caramia\y_encoder.pkl")
scalar = joblib.load(r"D:\anaco\caramia\scale_thy.pkl")

# Load recommendations
with open(r"D:\thyroid\treatment_recommendations.json", "r") as f:
    recommendations = json.load(f)

# Define the exact feature order based on your input_data example
feature_order = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 
    'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'referral_source'
]

# Categorical columns (for encoding)
categorical_cols = [
    'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_meds', 'sick',
    'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid', 
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 
    'referral_source'
]

# Continuous columns (for conversion to float)
continuous_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']

def get_age_group(age):
    """Categorize age into predefined age groups."""
    if age < 18:
        return None  # Return None for under 18
    elif age < 31:
        return "18-30"
    elif age < 51:
        return "31-50"
    elif age < 71:
        return "51-70"
    else:
        return "70+"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        sex = data.get("sex", "Male").strip().capitalize()
        age = float(data.get("age", 30))
        age_group = get_age_group(age)

        if age_group is None:
            return render_template('result.html', 
                                   prediction="No recommendations for age under 18.", 
                                   recommendation={})

        # Encode categorical values
        for col in categorical_cols:
            if col in data and col in label_encoders:
                if data[col] in label_encoders[col].classes_:
                    data[col] = label_encoders[col].transform([data[col]])[0]
                else:
                    data[col] = -1
            elif col not in data:
                data[col] = -1

        # Convert continuous values
        for col in continuous_cols:
            if col in data:
                try:
                    data[col] = float(data[col])
                except ValueError:
                    data[col] = 0
            elif col not in data:
                data[col] = 0

        input_data = [data.get(col, 0) for col in feature_order]
        input_data = np.array(input_data).reshape(1, -1)
        input_data_scaled = scalar.transform(input_data)

        # Get prediction probabilities
        probs = model.predict_proba(input_data_scaled)[0]
        print("Probabilities:", probs)

        sorted_indices = np.argsort(probs)[::-1]
        top1, top2 = sorted_indices[0], sorted_indices[1]
        top1_label = y_encoder.inverse_transform([top1])[0]
        top2_label = y_encoder.inverse_transform([top2])[0]

        negative_index = 2  # "Negative" is always at index 2
        negative_prob = probs[negative_index]

        # Updated logic for "Negative" prediction handling
        if top1 == negative_index:
            if negative_prob >= 0.7:
                y_pred = "Negative"
            else:
                y_pred = f"Negative — but possible early signs of {top2_label}"
        else:
            y_pred = y_encoder.inverse_transform([top1])[0]

        # Get recommendation based on prediction
        rec = (
            recommendations.get(y_pred.split(" —")[0], {})
                           .get(sex, {})
                           .get(age_group, {"Error": "No recommendations available."})
        )

        return render_template('result.html', prediction=y_pred, recommendation=rec)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}", recommendation={})

if __name__ == '__main__':
    app.run(debug=True)
