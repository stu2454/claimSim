from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
import os
import numpy as np
import random

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for development/testing

# Set up upload folder for storing uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to extract parameters
def extract_parameters(data):
    parameters = {}
    # Payment Amount statistics
    parameters['Payment Amount Mean'] = data['Payment Amount'].mean()
    parameters['Payment Amount Std'] = data['Payment Amount'].std()
    parameters['Payment Amount Min'] = data['Payment Amount'].min()
    parameters['Payment Amount Max'] = data['Payment Amount'].max()

    # Benchmark statistics
    parameters['Benchmark Mean'] = data['Benchmark'].mean()
    parameters['Benchmark Std'] = data['Benchmark'].std()
    parameters['Benchmark Min'] = data['Benchmark'].min()
    parameters['Benchmark Max'] = data['Benchmark'].max()

    # Percentage of claims exceeding the benchmark
    exceeding_claims = data[data['Payment Amount'] > data['Benchmark']].copy()
    exceeding_claims['Excess Amount'] = exceeding_claims['Payment Amount'] - exceeding_claims['Benchmark']
    parameters['Percentage Exceeding Benchmark'] = (len(exceeding_claims) / len(data)) * 100

    # Excess amounts for claims exceeding benchmark
    parameters['Excess Amount Mean'] = exceeding_claims['Excess Amount'].mean()
    parameters['Excess Amount Std'] = exceeding_claims['Excess Amount'].std()
    parameters['Excess Amount Min'] = exceeding_claims['Excess Amount'].min()
    parameters['Excess Amount Max'] = exceeding_claims['Excess Amount'].max()

    # Claims per day analysis
    try:
        data['Payment Date'] = pd.to_datetime(data['Payment Date'], errors='coerce')  # Relaxed parsing
        if data['Payment Date'].isnull().any():
            print(f"Warning: {data['Payment Date'].isnull().sum()} invalid dates found and set to NaT.")
    except Exception as e:
        raise ValueError(f"Date parsing failed: {str(e)}")

    claims_per_day = data.groupby(['Anonymised_participant', 'Payment Date']).size().reset_index(name='Claim Count')
    multiple_claims = claims_per_day[claims_per_day['Claim Count'] > 1]
    parameters['Multiple Claims Proportion'] = (multiple_claims['Anonymised_participant'].nunique() /
                                                 data['Anonymised_participant'].nunique()) * 100
    parameters['Multiple Claims Mean'] = multiple_claims['Claim Count'].mean()
    parameters['Multiple Claims Std'] = multiple_claims['Claim Count'].std()
    parameters['Multiple Claims Min'] = multiple_claims['Claim Count'].min()
    parameters['Multiple Claims Max'] = multiple_claims['Claim Count'].max()

    return parameters

# Simulation class
class Simulation:
    def __init__(self, parameters, start_date, end_date, n_participants=100, n_providers=20):
        self.parameters = parameters
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.participants = [f"P{str(i).zfill(4)}" for i in range(1, n_participants + 1)]
        self.providers = [f"PR{str(i).zfill(3)}" for i in range(1, n_providers + 1)]
        self.total_days = (self.end_date - self.start_date).days + 1

    def run(self, n_claims=1000, multiple_claim_proportion=10, max_claims_per_day=10):
        claims = []
        np.random.seed(42)
        random.seed(42)

        # Number of participants with multiple claims
        n_multiple_claim_participants = int(len(self.participants) * multiple_claim_proportion / 100)
        multiple_claim_participants = random.sample(self.participants, n_multiple_claim_participants)

        for _ in range(n_claims):
            participant = random.choice(self.participants)
            provider = random.choice(self.providers)
            claim_date = self.start_date + pd.Timedelta(days=random.randint(0, self.total_days - 1))
            unit_price = self._generate_unit_price()
            quantity = self._generate_quantity()
            payment_amount = unit_price * quantity
            benchmark = self.parameters.get('Benchmark Mean', 1000)

            claims.append({
                "Anonymised_participant": participant,
                "Provider ID": provider,
                "Payment Date": claim_date.strftime('%Y-%m-%d'),
                "Unit Price": round(unit_price, 2),
                "Quantity": quantity,
                "Payment Amount": round(payment_amount, 2),
                "Benchmark": round(benchmark, 2),
                "Exceeds Benchmark": payment_amount > benchmark
            })

        df = pd.DataFrame(claims)

        # Simulate multiple claims for selected participants
        for participant in multiple_claim_participants:
            days_to_claim = random.sample(range(self.total_days), max_claims_per_day)
            for day in days_to_claim:
                claim_date = self.start_date + pd.Timedelta(days=day)
                for _ in range(random.randint(2, max_claims_per_day)):  # Simulate 2 to max_claims_per_day claims
                    unit_price = self._generate_unit_price()
                    quantity = self._generate_quantity()
                    payment_amount = unit_price * quantity
                    benchmark = self.parameters.get('Benchmark Mean', 1000)

                    claims.append({
                        "Anonymised_participant": participant,
                        "Provider ID": random.choice(self.providers),
                        "Payment Date": claim_date.strftime('%Y-%m-%d'),
                        "Unit Price": round(unit_price, 2),
                        "Quantity": quantity,
                        "Payment Amount": round(payment_amount, 2),
                        "Benchmark": round(benchmark, 2),
                        "Exceeds Benchmark": payment_amount > benchmark
                    })

        return pd.DataFrame(claims)

    def _generate_unit_price(self):
        mean = self.parameters.get('Unit Price Mean', 500)
        std = self.parameters.get('Unit Price Std', 100)
        return max(0, np.random.normal(mean, std))

    def _generate_quantity(self):
        mean = self.parameters.get('Quantity Mean', 1)
        std = self.parameters.get('Quantity Std', 1)
        return max(1, int(np.random.normal(mean, std)))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the file is included in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Validate that the uploaded file is a .csv
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only .csv files are supported"}), 400

        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load the data and preprocess
        data = pd.read_csv(file_path)
        data['Payment Amount'] = data['Payment Amount'].replace('[\\$,]', '', regex=True).astype(float)
        data['Benchmark'] = data['Benchmark'].replace(',', '', regex=True).astype(float)

        # Extract parameters using the existing function
        parameters = extract_parameters(data)

        # Convert parameters to native Python types for JSON serialization
        parameters = {key: (int(value) if isinstance(value, np.int64) else
                            float(value) if isinstance(value, np.float64) else value)
                      for key, value in parameters.items()}

        return jsonify({"parameters": parameters})

    except Exception as e:
        # Handle any errors during processing
        return jsonify({"error": f"Failed to process the file: {str(e)}"}), 500

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        parameters = data.get('parameters')
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2023-12-31')
        n_participants = int(data.get('n_participants', 100))
        n_providers = int(data.get('n_providers', 20))
        n_claims = int(data.get('n_claims', 1000))
        multiple_claim_proportion = float(data.get('multiple_claim_proportion', 10))
        max_claims_per_day = int(data.get('max_claims_per_day', 10))

        if not parameters:
            return jsonify({"error": "Parameters not provided. Please upload a dataset first."}), 400

        sim = Simulation(parameters, start_date, end_date, n_participants, n_providers)
        simulated_data = sim.run(n_claims, multiple_claim_proportion, max_claims_per_day)

        claims_exceeding = simulated_data[simulated_data['Exceeds Benchmark']].shape[0]
        output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'simulated_data.csv')
        simulated_data.to_csv(output_csv_path, index=False)

        return jsonify({
            "message": "Simulation completed successfully.",
            "csv_file_path": output_csv_path,
            "data_preview": simulated_data.head(10).to_dict(orient='records'),
            "total_claims_exceeding_benchmark": claims_exceeding
        })

    except Exception as e:
        return jsonify({"error": f"Failed to run simulation: {str(e)}"}), 500

@app.route('/download', methods=['GET'])
def download_csv():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'simulated_data.csv')
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found."}), 404

if __name__ == '__main__':
    app.run(debug=True)

