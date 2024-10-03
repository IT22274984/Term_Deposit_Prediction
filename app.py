from flask import Flask, request, render_template, redirect, url_for
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Load the trained model
with open('gradient_boosting_model.pkl', 'rb') as model_file:
    model_pipeline = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = {
        'age': int(request.form['age']),
        'balance': float(request.form['balance']),
        'day': int(request.form['day']),
        'campaign': int(request.form['campaign']),
        'duration': int(request.form['duration']),
        'job': request.form['job'],
        'marital': request.form['marital'],
        'education': request.form['education'],
        'default': request.form['default'],
        'housing': request.form['housing'],
        'loan': request.form['loan'],
        'contact': request.form['contact'],
        'month': request.form['month'],
        'poutcome': request.form['poutcome']
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model_pipeline.predict(input_df)
    result = "Yes" if prediction[0] == 1 else "No"
    
    # Load the original dataset
    test_data = pd.read_csv('deposit term dirty null.csv')
    test_data['y'] = test_data['y'].map({'yes': 1, 'no': 0})
    
    # Calculate subscription probability by month
    monthly_trend = test_data.groupby('month')['y'].mean().reset_index()
    monthly_trend.columns = ['month', 'subscription_probability']

    # Map month names to numbers for proper sorting
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    monthly_trend['month_num'] = monthly_trend['month'].map(month_map)

    # Sort the data by month number
    monthly_trend = monthly_trend.sort_values('month_num')

    # Generate the probability trend plot
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_trend['month'], monthly_trend['subscription_probability'], marker='o')
    plt.title('Subscription Probability Trend Over Time')
    plt.xlabel('Month')
    plt.ylabel('Average Subscription Probability')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert the trend plot to PNG in-memory and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    trend_graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    # Redirect to the result page with data
    return render_template('result.html', result=result, trend_graph_base64=trend_graph_base64)


@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)