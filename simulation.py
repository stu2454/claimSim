import pandas as pd
import numpy as np


# Extract parameters from the dataset
def extract_parameters(data):
    # Clean numeric columns
    for col in ['Payment Amount', 'Benchmark', 'Unit Price', 'Quantity']:
        if col in data.columns:
            data[col] = data[col].replace('[\$,]', '', regex=True).replace(',', '', regex=True)
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop rows with missing critical data
    data = data.dropna(subset=['Payment Amount', 'Benchmark'])

    # Extract Quantity statistics if the column exists
    if 'Quantity' in data.columns and not data['Quantity'].isna().all():
        quantity_mean = data['Quantity'].mean()
        quantity_std = data['Quantity'].std()
    else:
        quantity_mean = None
        quantity_std = None

    # Proportion of $1 Unit Price claims
    if 'Unit Price' in data.columns and 'Quantity' in data.columns:
        one_dollar_claims = data[data['Unit Price'] == 1]
        non_one_dollar_claims = data[data['Unit Price'] != 1]

        proportion_one_dollar = len(one_dollar_claims) / len(data)
        one_dollar_quantity_mean = one_dollar_claims['Quantity'].mean() if not one_dollar_claims.empty else None
        one_dollar_quantity_std = one_dollar_claims['Quantity'].std() if not one_dollar_claims.empty else None

        non_one_dollar_unit_price_mean = non_one_dollar_claims['Unit Price'].mean() if not non_one_dollar_claims.empty else None
        non_one_dollar_unit_price_std = non_one_dollar_claims['Unit Price'].std() if not non_one_dollar_claims.empty else None
        non_one_dollar_quantity_mean = non_one_dollar_claims['Quantity'].mean() if not non_one_dollar_claims.empty else None
        non_one_dollar_quantity_std = non_one_dollar_claims['Quantity'].std() if not non_one_dollar_claims.empty else None
    else:
        proportion_one_dollar = 0
        one_dollar_quantity_mean = None
        one_dollar_quantity_std = None

        non_one_dollar_unit_price_mean = None
        non_one_dollar_unit_price_std = None
        non_one_dollar_quantity_mean = None
        non_one_dollar_quantity_std = None

    # Proportion of claims exceeding benchmark
    claims_over_benchmark = data[data['Payment Amount'] > data['Benchmark']]
    proportion_over_benchmark = len(claims_over_benchmark) / len(data)

    # Excess payment statistics
    if not claims_over_benchmark.empty:
        excess_amounts = claims_over_benchmark['Payment Amount'] - claims_over_benchmark['Benchmark']
        excess_mean = excess_amounts.mean()
        excess_std = excess_amounts.std()
    else:
        excess_mean = None
        excess_std = None

    # Proportion of participants with multiple claims per day
    if 'Anonymised_participant' in data.columns:
        claims_per_day = data.groupby(['Anonymised_participant', 'Payment Date']).size()
        multiple_claims_proportion = (claims_per_day > 1).mean()
    else:
        multiple_claims_proportion = None

    # Return the calculated parameters
    parameters = pd.DataFrame({
        'Metric': [
            'Payment Amount Mean', 'Payment Amount Std', 'Payment Amount Min', 'Payment Amount Max',
            'Benchmark Mean', 'Benchmark Std', 'Benchmark Min', 'Benchmark Max',
            'Proportion Over Benchmark', 'Excess Amount Mean', 'Excess Amount Std',
            'Proportion Multiple Claims', 'Quantity Mean', 'Quantity Std',
            'Proportion $1 Unit Price Claims', 'Quantity Mean ($1 Unit Price)', 'Quantity Std ($1 Unit Price)',
            'Unit Price Mean (Non-$1)', 'Unit Price Std (Non-$1)',
            'Quantity Mean (Non-$1)', 'Quantity Std (Non-$1)'
        ],
        'Value': [
            data['Payment Amount'].mean(), data['Payment Amount'].std(), data['Payment Amount'].min(),
            data['Payment Amount'].max(), data['Benchmark'].mean(), data['Benchmark'].std(),
            data['Benchmark'].min(), data['Benchmark'].max(),
            proportion_over_benchmark, excess_mean, excess_std,
            multiple_claims_proportion, quantity_mean, quantity_std,
            proportion_one_dollar, one_dollar_quantity_mean, one_dollar_quantity_std,
            non_one_dollar_unit_price_mean, non_one_dollar_unit_price_std,
            non_one_dollar_quantity_mean, non_one_dollar_quantity_std
        ]
    })

    return parameters



def simulate_and_extract(parameters_df, n_providers, n_claims=1000):
    parameters = parameters_df.set_index('Metric')['Value'].to_dict()

    np.random.seed(42)

    # Generate providers and assign participants
    providers = [f'PR{i:03}' for i in range(1, n_providers + 1)]
    participants = [f'P{i:03}' for i in range(1, 101)]

    provider_to_participants = {}
    for provider in providers:
        num_participants = np.random.randint(1, 10)  # Simulate 1-10 participants per provider
        provider_to_participants[provider] = np.random.choice(participants, num_participants, replace=False)

    # Separate proportions for $1 and non-$1 claims
    one_dollar_claim_count = int(parameters.get('Proportion $1 Unit Price Claims', 0) * n_claims)
    non_one_dollar_claim_count = n_claims - one_dollar_claim_count

    # Simulate $1 claims
    one_dollar_data = pd.DataFrame({
        'Participant': np.random.choice(participants, size=one_dollar_claim_count),
        'Payment Date': np.random.choice(pd.date_range(start='2023-01-01', end='2023-12-31'), size=one_dollar_claim_count),
        'Unit Price': 1,
        'Quantity': np.random.normal(
            loc=parameters.get('Quantity Mean ($1 Unit Price)', 1000),
            scale=parameters.get('Quantity Std ($1 Unit Price)', 200),
            size=one_dollar_claim_count
        ).clip(1, 5000).round().astype(int),
        'Benchmark': parameters.get('Benchmark Mean', 10000)  # Use real-world mean or default
    })

    # Simulate non-$1 claims
    non_one_dollar_data = pd.DataFrame({
        'Participant': np.random.choice(participants, size=non_one_dollar_claim_count),
        'Payment Date': np.random.choice(pd.date_range(start='2023-01-01', end='2023-12-31'), size=non_one_dollar_claim_count),
        'Unit Price': np.random.normal(
            loc=parameters.get('Unit Price Mean (Non-$1)', 100),
            scale=parameters.get('Unit Price Std (Non-$1)', 50),
            size=non_one_dollar_claim_count
        ).clip(
            parameters.get('Unit Price Min', 10), 
            parameters.get('Unit Price Max', 12359.76)
        ).round(2),
        'Quantity': np.random.normal(
            loc=parameters.get('Quantity Mean (Non-$1)', 5),
            scale=parameters.get('Quantity Std (Non-$1)', 2),
            size=non_one_dollar_claim_count
        ).clip(1, 50).round().astype(int),
        'Benchmark': np.random.normal(
            loc=parameters.get('Benchmark Mean', 10000),
            scale=parameters.get('Benchmark Std', 2000),
            size=non_one_dollar_claim_count
        ).round(2)
    })

    # Combine $1 and non-$1 claims
    simulated_data = pd.concat([one_dollar_data, non_one_dollar_data], ignore_index=True)

    simulated_data['Provider'] = simulated_data['Participant'].map(
        lambda p: next(
            (provider for provider, plist in provider_to_participants.items() if p in plist),
            np.random.choice(providers)
        )
    )

    # Add Anonymised_participant column
    simulated_data['Anonymised_participant'] = simulated_data['Participant']

    # Calculate Payment Amount as Unit Price × Quantity
    simulated_data['Payment Amount'] = (simulated_data['Unit Price'] * simulated_data['Quantity']).round(2)

    return simulated_data, parameters_df




