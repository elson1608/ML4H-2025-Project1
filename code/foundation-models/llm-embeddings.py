from pathlib import Path
import pandas as pd
import ollama

script_dir = Path(__file__).parent

descriptions = {
  "Albumin": "Albumin",
  "ALP": "Alkaline phosphatase",
  "ALT": "Alanine transaminase",
  "AST": "Aspartate transaminase",
  "Bilirubin": "Bilirubin",
  "BUN": "Blood urea nitrogen",
  "Cholesterol": "Cholesterol",
  "Creatinine": "Serum creatinine",
  "DiasABP": "Invasive diastolic arterial blood pressure",
  "FiO2": "Fractional inspired O2",
  "GCS": "Glasgow Coma Score",
  "Glucose": "Serum glucose",
  "HCO3": "Serum bicarbonate",
  "HCT": "Hematocrit",
  "HR": "Heart rate",
  "K": "Serum potassium",
  "Lactate": "Lactate",
  "Mg": "Serum magnesium",
  "MAP": "Invasive mean arterial blood pressure",
  "MechVent": "Mechanical ventilation respiration",
  "Na": "Serum sodium",
  "NIDiasABP": "Non-invasive diastolic arterial blood pressure",
  "NIMAP": "Non-invasive mean arterial blood pressure",
  "NISysABP": "Non-invasive systolic arterial blood pressure",
  "PaCO2": "partial pressure of arterial CO2",
  "PaO2": "Partial pressure of arterial O2",
  "pH": "Arterial pH",
  "Platelets": "Platelets",
  "RespRate": "Respiration rate",
  "SaO2": "O2 saturation in hemoglobin",
  "SysABP": "Invasive systolic arterial blood pressure",
  "Temp": "°C",
  "TroponinI": "Troponin-I",
  "TroponinT": "Troponin-T",
  "Urine": "Urine output",
  "WBC": "White blood cell count",
  "Weight": "Weight",
  "Weight(static)": "Weight [measured at arrival]",
  "Height": "Height",
  "Gender": "Gender",
  "Age": "Age"
}

units = {
  "Albumin": "g/dL",
  "ALP": "IU/L",
  "ALT": "IU/L",
  "AST": "IU/L",
  "Bilirubin": "mg/dL",
  "BUN": "mg/dL",
  "Cholesterol": "mg/dL",
  "Creatinine": "mg/dL",
  "DiasABP": "mmHg",
  "FiO2": "",
  "GCS": "",
  "Glucose": "mg/dL",
  "HCO3": "mmol/L",
  "HCT": "%",
  "HR": "bpm",
  "K": "mEq/L",
  "Lactate": "mmol/L",
  "Mg": "mmol/L",
  "MAP": "mmHg",
  "MechVent": "",
  "Na": "mEq/L",
  "NIDiasABP": "mmHg",
  "NIMAP": "mmHg",
  "NISysABP": "mmHg",
  "PaCO2": "mmHg",
  "PaO2": "mmHg",
  "pH": "",
  "Platelets": "(cells/nL)",
  "RespRate": "bpm",
  "SaO2": "%",
  "SysABP": "mmHg",
  "Temp": "°C",
  "TroponinI": "μg/L",
  "TroponinT": "μg/L",
  "Urine": "mL",
  "WBC": "cells/nL",
  "Weight": "kg",
  "Weight(static)": "kg",
  "Height": "cm",
  "Gender": "",
  "Age": "years"  
}


aggregations = {
  'min': 'a minimum',
  'max': 'a maximum',
  'mean': 'an average'
}


def compute_simple_features(df):
    static_cols = ['Height', 'Age', 'Gender', 'Weight(static)']
    dynamic_cols = [column for column in df.columns if column not in ['RecordID', 'Time', 'Height', 'Age', 'Gender', 'Weight(static)']]
    
    non_aggregated = df[static_cols].groupby(df['RecordID']).first()
    aggregated = df.groupby('RecordID')[dynamic_cols].agg([
        'mean',
        # 'median',
        'min',
        'max',
        # 'std',
        # 'first',
        # 'last',
        # ('rate_of_change', lambda x: (x.iloc[-1] - x.iloc[0]) / 49)  # Named tuple approach
    ])
    aggregated.columns = ['_'.join(col) for col in aggregated.columns]
    df = pd.concat([non_aggregated, aggregated], axis=1)
    
    return df





def prompt_llm(patient):
  prompt = "The patient has "

  for param, value in patient.items():
    names = param.split('_', 1)
    feature = names[0]
    description = descriptions[feature]
    if len(names) > 1:
      aggregation = names[1]
      prompt += f"""{aggregations[aggregation]} {description} of {value}"""
      if units[feature]:
        prompt += f""" {units[feature]}"""
      
      if feature == 'TroponinT' and aggregation == 'max':
        prompt += f"""."""
      else:
        prompt += f""", """
    else:
      if feature == "Gender":
        prompt += f"""{feature} of {'female' if value == 0 else 'male'}, """
      else:
        prompt += f"""{feature} of {value} {units[feature]}, """
       
    
    
  
  # print(prompt)
  
  response = ollama.embed(
    model='gemma2:2b',
    input=prompt,
  )

  embedding = response['embeddings'][0]
  
  return embedding


for set_type in ['a', 'c']:
  
  df = pd.read_parquet(script_dir / f'../../data/set-{set_type}-imputed.parquet').sort_values(by=['RecordID','Time'])
  df = compute_simple_features(df)
  df = df.round(2)
  
  print(f"Starting with set {set_type}")
  
  data_dict = {
   "RecordID": [],
   "Embedding": []
  }
    
  i = 1
  for record_id, patient in df.iterrows():
    print(i)
    embedding = prompt_llm(patient)
    data_dict["RecordID"].append(record_id) 
    data_dict["Embedding"].append(embedding)
    i += 1

  llm_df = pd.DataFrame(data_dict)
  llm_df.to_parquet(script_dir / f"../../data/set-{set_type}-foundation-model-embeddings.parquet", index=False)






