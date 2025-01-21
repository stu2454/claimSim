from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import io
import json
from simulation import extract_parameters, simulate_and_extract

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check for file in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Load the file into a DataFrame
        data = pd.read_csv(file)

        # Extract parameters using simulation.py
        parameters_df = extract_parameters(data)

        # Return parameters as JSON
        return jsonify({'message': 'File uploaded successfully', 'parameters': parameters_df.to_dict(orient='records')}), 200
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/simulate', methods=['POST'])
def simulate_data():
    try:
        # Parse parameters
        parameters_json = request.form.get('parameters')
        if not parameters_json:
            raise ValueError("No parameters provided")

        parameters_df = pd.DataFrame(json.loads(parameters_json))
        num_claims = int(request.form.get('n_claims', 1000))
        num_providers = int(request.form.get('n_providers', 20))

        # Run the simulation
        simulated_data, real_world_params_df = simulate_and_extract(parameters_df, num_providers, num_claims)

        # Analyze the simulated data
        simulated_metrics = extract_parameters(simulated_data)

        # Add simulated metrics and comparisons to the real-world parameters sheet
        real_world_params_df['Simulated Value'] = real_world_params_df['Metric'].map(
            lambda metric: simulated_metrics.loc[simulated_metrics['Metric'] == metric, 'Value'].values[0]
            if metric in simulated_metrics['Metric'].values else None
        )
        real_world_params_df['Difference'] = abs(real_world_params_df['Value'] - real_world_params_df['Simulated Value'])

        # Create an Excel file with two sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            simulated_data.to_excel(writer, index=False, sheet_name='Simulated Data')
            real_world_params_df.to_excel(writer, index=False, sheet_name='Real-World Parameters')

        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='simulated_data_comparison.xlsx'
        )
    except Exception as e:
        print("Unexpected error in /simulate:", e)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(debug=True)
