# claimSim

claimSim is a web-based application designed to simulate and analyse claims for Assistive Technology. It provides users with an intuitive interface to input data, run simulations, and view results, facilitating better understanding and management of insurance claim processes.

## Features

- **User-Friendly Interface**: Interact with the application through a clean and responsive web interface.
- **Simulation Engine**: Run detailed simulations of AT claims based on user-defined parameters.
- **Data Visualisation**: View simulation results through comprehensive charts and tables.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/stu2454/claimSim.git
   cd claimSim
   ```

2. **Set Up a Python Virtual Environment** (recommended):
   
   Create and activate a virtual environment to isolate the project's dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   Install all required Python packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   The application requires specific environment variables to function correctly. Create a `.env` file in the project root and specify the following variables:
   ```env
   FLASK_APP=main.py
   FLASK_ENV=development
   SECRET_KEY=<your_secret_key>
   DATABASE_URL=<your_database_url>
   ```

   Replace `<your_secret_key>` and `<your_database_url>` with appropriate values. For local development, you can use a SQLite database or connect to a PostgreSQL instance.

## Usage

1. **Start the Application**:
   Run the following command to start the Flask development server:
   ```bash
   python main.py
   ```

2. **Access the Web Interface**:
   Open your web browser and navigate to `http://localhost:5000` to interact with the application.

## Project Structure

- **`main.py`**: Initialises the Flask web server and defines the main routes.
- **`app.py`**: Contains the core application logic, configurations, and setups.
- **`simulate.py`**: Implements the simulation logic and data generation processes.
- **`templates/index.html`**: Provides the front-end user interface for the application.
- **`static/`**: Contains static files such as CSS, JavaScript, and images.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request for any feature additions or bug fixes.

## Licence

This project is licensed under the MIT Licence. See the `LICENCE` file for details.


