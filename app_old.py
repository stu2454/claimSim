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
    exceeding_claims = data[data['Payment Amount'] > data['Benchmark']]
    parameters['Percentage Exceeding Benchmark'] = (len(exceeding_claims) / len(data)) * 100

    # Excess amounts for claims exceeding benchmark
    exceeding_claims['Excess Amount'] = exceeding_claims['Payment Amount'] - exceeding_claims['Benchmark']
    parameters['Excess Amount Mean'] = exceeding_claims['Excess Amount'].mean()
    parameters['Excess Amount Std'] = exceeding_claims['Excess Amount'].std()
    parameters['Excess Amount Min'] = exceeding_claims['Excess Amount'].min()
    parameters['Excess Amount Max'] = exceeding_claims['Excess Amount'].max()

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

    def run(self, n_claims=1000):
        claims = []
        np.random.seed(42)
        random.seed(42)

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

        # Adjust the proportion of claims exceeding the benchmark
        real_proportion = self.parameters.get('Percentage Exceeding Benchmark', 4) / 100
        n_exceeding = int(n_claims * real_proportion)

        if n_exceeding > 0:
            non_exceeding_idx = df[df['Exceeds Benchmark'] == False].index
            if len(non_exceeding_idx) >= n_exceeding:
                selected_indices = random.sample(list(non_exceeding_idx), n_exceeding)
                for idx in selected_indices:
                    # Generate realistic excess amounts
                    excess_amount = max(
                        min(np.random.normal(self.parameters.get('Excess Amount Mean', 200),
                                             self.parameters.get('Excess Amount Std', 50)),
                            self.parameters.get('Excess Amount Max', 350)),
                        self.parameters.get('Excess Amount Min', 5)
                    )
                    df.loc[idx, 'Payment Amount'] = df.loc[idx, 'Benchmark'] + excess_amount
                    df.loc[idx, 'Exceeds Benchmark'] = True

        return df

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
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only .csv files are supported"}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        data = pd.read_csv(file_path)
        data['Payment Amount'] = data['Payment Amount'].replace('[\$,\u20AC,\u00A3]', '', regex=True).astype(float)
        data['Benchmark'] = data['Benchmark'].replace(',', '', regex=True).astype(float)

        parameters = extract_parameters(data)
        return jsonify({"parameters": parameters})

    except Exception as e:
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

        if not parameters:
            return jsonify({"error": "Parameters not provided. Please upload a dataset first."}), 400

        sim = Simulation(parameters, start_date, end_date, n_participants, n_providers)
        simulated_data = sim.run(n_claims)

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

