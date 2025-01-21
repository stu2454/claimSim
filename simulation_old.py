import pandas as pd
import numpy as np
import random
from tkinter import Tk, filedialog

# GUI to select the real-world data file
def select_real_world_data_file():
    Tk().withdraw()  # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select the real-world data file",
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

# Extract parameters from the real-world dataset
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
    parameters['Percentage Exceeding Benchmark'] = (
        (data['Payment Amount'] > data['Benchmark']).mean() * 100
    )

    # Excess amounts for claims exceeding benchmark
    exceeding_claims = data[data['Payment Amount'] > data['Benchmark']]
    parameters['Excess Amount Mean'] = (exceeding_claims['Payment Amount'] - exceeding_claims['Benchmark']).mean()
    parameters['Excess Amount Std'] = (exceeding_claims['Payment Amount'] - exceeding_claims['Benchmark']).std()

    # Unit Price and Quantity statistics (if available)
    if 'Unit Price' in data.columns and 'Quantity' in data.columns:
        parameters['Unit Price Mean'] = data['Unit Price'].mean()
        parameters['Unit Price Std'] = data['Unit Price'].std()
        parameters['Quantity Mean'] = data['Quantity'].mean()
        parameters['Quantity Std'] = data['Quantity'].std()

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
            benchmark = self.parameters['Benchmark Mean']
            claims.append({
                "Anonymised_participant": participant,
                "Provider ID": provider,
                "Payment Date": claim_date,
                "Benchmark": f"${benchmark:.2f}",
                "Unit Price": f"${unit_price:.2f}",
                "Quantity": quantity,
                "Payment Amount": f"${payment_amount:.2f}",
                "Exceeds Benchmark": payment_amount > benchmark
            })

        return pd.DataFrame(claims)

    def _generate_unit_price(self):
        mean = self.parameters.get('Unit Price Mean', 500)
        std = self.parameters.get('Unit Price Std', 100)
        return max(0, round(np.random.normal(mean, std), 2))

    def _generate_quantity(self):
        mean = self.parameters.get('Quantity Mean', 1)
        std = self.parameters.get('Quantity Std', 1)
        return max(1, int(round(np.random.normal(mean, std), 0)))

# Main program
if __name__ == "__main__":
    # Step 1: Select the real-world data file
    file_path = select_real_world_data_file()
    if not file_path:
        print("No file selected. Exiting...")
        exit()

    # Step 2: Load and clean the dataset
    data = pd.read_csv(file_path)
    data['Payment Amount'] = data['Payment Amount'].replace('[\$,\u20AC,\u00A3]', '', regex=True).astype(float)
    data['Benchmark'] = data['Benchmark'].replace(',', '', regex=True).astype(float)
    if 'Unit Price' in data.columns and 'Quantity' in data.columns:
        data['Unit Price'] = data['Unit Price'].replace('[\$,\u20AC,\u00A3]', '', regex=True).astype(float)
        data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')

    # Step 3: Extract parameters
    parameters = extract_parameters(data)
    print("\nExtracted Parameters:")
    for key, value in parameters.items():
        print(f"{key}: {value}")

    # Step 4: Run the simulation
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    sim = Simulation(parameters, start_date, end_date, n_participants=100, n_providers=20)
    simulated_data = sim.run(n_claims=1000)

    # Step 5: Save the simulated data
    output_file = "simulated_data.csv"
    simulated_data.to_csv(output_file, index=False)
    print(f"\nSimulated data saved to {output_file}")
    print(simulated_data.head())

