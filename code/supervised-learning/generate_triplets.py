from pathlib import Path
import pandas as pd
import json
import warnings
import numpy as np
warnings.simplefilter(action='ignore', category=FutureWarning)

script_dir = Path(__file__).parent

with open(script_dir / "scaling_params_dict.json", "r") as f:
    scaling_params = json.load(f)
# with open("one_hot_encodings.json", "r") as f:
#     one_hot_dict = json.load(f)

with open(script_dir / "sensor_to_number.json", "r") as f:
    sensor_to_number_dict = json.load(f)
def scale_parameter(row, scaling_params):
    """
    Scales a parameter value using the mean and variance from the scaling_params dictionary.

    Args:
        row (pd.Series): A row of the DataFrame containing 'Sensor' and 'Value'.
        scaling_params (dict): A dictionary with scaling parameters (mean and variance).

    Returns:
        float: The scaled value.
    """
    sensor = row['Sensor']
    value = row['Value']

    if sensor in scaling_params:
        mean = scaling_params[sensor]['mean']
        std_dev = np.sqrt(scaling_params[sensor]['variance'])
        return (value - mean) / std_dev if std_dev > 0 else value  # Avoid division by zero
    return value  # Return the original value if the sensor is not in the scaling_params

def scale_timestamp_to_unit_range(timestamp: str) -> float:
    """
    Scales a timestamp string in the format 'HH:MM' into a number in the range [0, 1],
    where '00:00' maps to 0 and '48:00' maps to 1.

    Args:
        timestamp (str): A string representing the timestamp in 'HH:MM' format.

    Returns:
        float: A scaled value in the range [0, 1].
    """
    # Split the timestamp into hours and minutes
    hours, minutes = map(int, timestamp.split(":"))
    
    # Convert the timestamp into total minutes
    total_minutes = hours * 60 + minutes
    
    # Scale the total minutes to the range [0, 1]
    scaled_value = total_minutes / (48 * 60)  # 48 hours = 48 * 60 minutes
    
    return scaled_value

def generate_triplets(df):
    """
    Generate triplets of (Time, Sensor, Value) for each RecordID.

    Args:
        df (pd.DataFrame): Input DataFrame with columns like 'Time', 'RecordID', and sensor values.

    Returns:
        pd.DataFrame: DataFrame containing triplets with columns ['RecordID', 'Time', 'Sensor', 'Value'].
    """
    triplets = []

    # Extract unique RecordIDs
    unique_record_ids = df['RecordID'].unique()

    # Add one static triplet for Gender and Age per RecordID
    for record_id in unique_record_ids:
        patient_data = df[df['RecordID'] == record_id]
        
        # Extract Gender and Age (assuming they are constant for each RecordID)
        gender = patient_data['Gender'].iloc[0] if 'Gender' in patient_data.columns else None
        age = patient_data['Age'].iloc[0] if 'Age' in patient_data.columns else None

        if age is not None and 'Age' in scaling_params:
            mean = scaling_params['Age']['mean']
            std_dev = np.sqrt(scaling_params['Age']['variance'])
            age = (age - mean) / std_dev if std_dev > 0 else age  # Avoid division by zero
        # Add static triplets for Gender and Age
        if gender is not None:
            triplets.append((record_id, 0, 'Gender', gender))  # Use Time = 0 for static triplets
        if age is not None:
            triplets.append((record_id, 0, 'Age', age))  # Use Time = 0 for static triplets

    # Generate dynamic triplets for each row in the DataFrame
    for _, row in df.iterrows():
        record_id = row['RecordID']
        time = scale_timestamp_to_unit_range(row['Time'])

        # Iterate through all columns except 'Time' and 'RecordID'
        for sensor, value in row.items():
            if sensor not in ['Time', 'RecordID', 'Gender', 'Age'] and pd.notnull(value):
                if sensor in scaling_params:
                    mean = scaling_params[sensor]['mean']
                    std_dev = np.sqrt(scaling_params[sensor]['variance'])
                    value = (value - mean) / std_dev if std_dev > 0 else value  # Avoid division by zero
                
                # Append the triplet (RecordID, Time, Sensor, Value)
                triplets.append((record_id, time, sensor, value))

    # Convert the triplets list to a DataFrame
    triplets_df = pd.DataFrame(triplets, columns=['RecordID', 'Time', 'Sensor', 'Value'])



    return triplets_df

# Define all expected columns
expected_columns = [
    'Age', 'Gender', 'Height', 'Weight', 'Albumin', 'ALP', 'ALT', 'AST',
    'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', 'FiO2', 'DiasABP',
    'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP',
    'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2',
    'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI',
    'TroponinT', 'Urine', 'WBC', 'RecordID'
]

# Define static parameters
static_params = ['Age','Gender','Height', 'Weight', 'RecordID']

# Get the list of sensor names
sensor_names = list(scaling_params.keys())

# Process each Parquet file
for patient_set in ['a', 'b', 'c']:
    # Load the parquet file
    df = pd.read_parquet(script_dir / f'../../data/set-{patient_set}.parquet')
    df = df.drop(columns=['ICUType'])
    print('df head', df.head())
    df = df.replace(-1.0, None)
    df['Time'] = df['Time'].astype(str)
                    ##outlier removal
    valid_ranges = {
        'Age': (0, 120),  # Age in years
        'Gender': (0, 1),  # Binary: 0 (Male), 1 (Female)
        'Height': (50, 250),  # Height in cm
        'Weight': (2, 300),  # Weight in kg
        'Albumin': (0.5, 6.0),  # Albumin in g/dL
        'ALP': (10, 2000),  # Alkaline phosphatase in U/L
        'ALT': (0, 2000),  # Alanine transaminase in U/L
        'AST': (0, 2000),  # Aspartate transaminase in U/L
        'Bilirubin': (0.0, 50.0),  # Bilirubin in mg/dL
        'BUN': (0, 300),  # Blood urea nitrogen in mg/dL
        'Cholesterol': (30, 1000),  # Cholesterol in mg/dL
        'Creatinine': (0.0, 20.0),  # Creatinine in mg/dL
        'FiO2': (0.21, 1.0),  # Fraction of inspired oxygen (21% to 100%)
        'DiasABP': (10, 150),  # Diastolic arterial blood pressure in mmHg
        'GCS': (3, 15),  # Glasgow Coma Scale score
        'Glucose': (10, 2000),  # Glucose in mg/dL
        'HCO3': (5, 50),  # Bicarbonate in mmol/L
        'HCT': (10, 80),  # Hematocrit in %
        'HR': (0, 300),  # Heart rate in bpm
        'K': (1.5, 10.0),  # Potassium in mmol/L
        'Lactate': (0.0, 30.0),  # Lactate in mmol/L
        'Mg': (0.0, 5.0),  # Magnesium in mmol/L
        'MAP': (20, 200),  # Mean arterial pressure in mmHg
        'MechVent': (0, 1),  # Binary: 0 (No), 1 (Yes)
        'Na': (100, 200),  # Sodium in mmol/L
        'NIDiasABP': (10, 150),  # Non-invasive diastolic blood pressure in mmHg
        'NIMAP': (20, 200),  # Non-invasive mean arterial pressure in mmHg
        'NISysABP': (30, 300),  # Non-invasive systolic blood pressure in mmHg
        'PaCO2': (10, 120),  # Partial pressure of carbon dioxide in mmHg
        'PaO2': (20, 800),  # Partial pressure of oxygen in mmHg
        'pH': (6.5, 8.0),  # Blood pH
        'Platelets': (0, 2000),  # Platelet count in 10^3/µL
        'RespRate': (0, 100),  # Respiratory rate in breaths per minute
        'SaO2': (0, 100),  # Oxygen saturation in %
        'SysABP': (30, 300),  # Systolic arterial blood pressure in mmHg
        'Temp': (25, 45),  # Temperature in °C
        'TroponinI': (0, 100),  # Troponin I in ng/mL
        'TroponinT': (0, 10),  # Troponin T in ng/mL
        'Urine': (0, 50000),  # Urine output in mL
        'WBC': (0.0, 500),  # White blood cell count in 10^3/µL
    }
    for param, (min_val, max_val) in valid_ranges.items():
        if param in df.columns:
            df[param] = df[param].apply(lambda x: x if min_val <= x <= max_val else None)
    # # Extract static variables including RecordID
    # static_vars = df[
    #     (df['Time'] == '00:00') & 
    #     (df['Parameter'].isin(static_params))
    # ].drop_duplicates(subset=['Parameter'], keep='first').set_index('Parameter')['Value']


    triplets = generate_triplets(df)
    triplets = triplets.dropna()
    triplets = triplets.sort_values(by=['RecordID', 'Time']).reset_index(drop=True)
    triplets['Sensor'] = triplets['Sensor'].map(sensor_to_number_dict)
            
            


    # Save the imputed DataFrame back to Parquet
    triplets.to_parquet(script_dir / f'../../data/set-{patient_set}-triplet.parquet', index=False)

    print(f"Saved imputed file: {script_dir / '../../data' / f'set-{patient_set}-triplet.parquet'}")


                
            