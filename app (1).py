from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the model from the same folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.save")
model = joblib.load(model_path)

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Glossary page
@app.route('/glossary')
def glossary():
    return render_template('glossary.html')

# Prediction page (example)
@app.route('/manual_predict', methods=['GET', 'POST'])
def manual_predict():
    if request.method == 'POST':
        try:
            # Example: assume your form sends features named 'feature1', 'feature2', etc.
            features = [float(x) for x in request.form.values()]
            prediction = model.predict([features])
            return render_template('output.html', prediction=prediction[0])
        except Exception as e:
            return f"Error: {e}"
    return render_template('manual_predict.html')  # Create this template for input form

if __name__ == "__main__":
    app.run(debug=True)
