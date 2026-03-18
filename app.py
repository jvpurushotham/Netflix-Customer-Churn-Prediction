# Importing the packages
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Loading the trained model
model = pickle.load(open('netflix_churn_model.pkl', 'rb'))

# List of features
FEATURES = [
    "age",
    "gender",
    "subscription_type",
    "watch_hours",
    "last_login_days",
    "region",
    "device",
    "monthly_fee",
    "payment_method",
    "number_of_profiles",
    "favorite_genre",
]

# mappings for categorical variables
gender_map = {"Female": 0, "Male": 1, "Other": 2}
subscription_map = {"Basic": 0, "Standard": 2, "Premium": 1}
device_map = {"Desktop": 0, "Laptop": 1, "Mobile": 2, "TV": 3, "Tablet": 4}
region_map = {"Africa":0, "Asia":1, "Europe":2, "North America":3, "Oceania":4, "South America":5}
payment_map = {"Credit Card":0, "Crypto":1, "Debit Card":2, "Gift Card":3, "PayPal":4}
genre_map = {"Action":0, "Comedy":1, "Documentary":2, "Drama":3, "Horror":4, "Romance":5, "Sci-Fi":6}

# Helper function to encode categorical values safely
def encode_feature(value, mapping):
    return mapping.get(value, 0) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form 
        input_features = [request.form[feature] for feature in FEATURES]

        # Convert numeric features to float 
        numeric_indices = [0, 3, 4, 7, 9]  # age, watch_hours, last_login_days, monthly_fee, number_of_profiles
        for i in numeric_indices:
            input_features[i] = float(input_features[i])

        # Encoding categorical variables
        input_features[1] = encode_feature(input_features[1], gender_map)
        input_features[2] = encode_feature(input_features[2], subscription_map)
        input_features[5] = encode_feature(input_features[5], region_map)
        input_features[6] = encode_feature(input_features[6], device_map)
        input_features[8] = encode_feature(input_features[8], payment_map)
        input_features[10] = encode_feature(input_features[10], genre_map)

        # Predicting churn and probability of churn 
        prediction = model.predict([input_features])[0]
        probability = model.predict_proba([input_features])[0][1]  # probability of churn

        result = "Likely to Churn" if prediction == 1 else "Not Likely to Churn"

        return render_template('index.html', result=result, probability=round(probability*100, 2))

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}", probability=None)
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
