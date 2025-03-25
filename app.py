from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import json

app = Flask(__name__)

# Load model and encoders
model = joblib.load(r"D:\anaco\caramia\xgb_model.pkl")
label_encoders = joblib.load(r"D:\anaco\caramia\label_encoders.pkl")
y_encoder = joblib.load(r"D:\anaco\caramia\y_encoder.pkl")

# Load recommendations
with open(r"D:\thyroid\treatment_recommendations.json", "r") as f:
    recommendations = json.load(f)

# Define categorical and continuous columns
categorical_cols = [
    "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds", "sick",
    "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid", 
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", 
    "psych", "referral_source"
]
continuous_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]

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
        sex = data.get("sex", "Male").strip().capitalize()  # Normalize input
        age = float(data.get("age", 30))  # Default to 30 if missing
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
                    data[col] = -1  # Default for unseen values

        # Convert continuous values
        for col in continuous_cols:
            if col in data:
                try:
                    data[col] = float(data[col])
                except ValueError:
                    data[col] = 0  # Default for invalid inputs

        # Prepare input
        input_data = np.array(list(data.values())).reshape(1, -1)
        prediction = model.predict(input_data)
        y_pred = y_encoder.inverse_transform(prediction)[0]

        # Fetch recommendation safely
        rec = (
            recommendations.get(y_pred, {})
                           .get(sex, {})
                           .get(age_group, {"Error": "No recommendations available."})
        )
        print(rec)
        return render_template('result.html', prediction=y_pred, recommendation=rec)
    
    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}", recommendation={})

if __name__ == '__main__':
    app.run(debug=True)
