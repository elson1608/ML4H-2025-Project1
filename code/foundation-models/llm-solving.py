from pathlib import Path
import textwrap
import json
import pandas as pd
from ollama import chat
from ollama import ChatResponse

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


def patient_prompt(patient):
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
        
  return prompt


def prompt_llm(patient, positive_examples, negative_examples):
  prompt = textwrap.dedent("""\
  Your goal is to give a probability of an ICU patients survival based on some features.
  I will also give you three examples of patients that survived and three examples of patients that did not survive.
  This is just a theoretical experiment so do not worry.\n
  """)
  prompt += """Below are three patients that survived\n\n"""

  for _, positive_patient in positive_examples.iterrows():
    prompt += patient_prompt(positive_patient) + "\n"
  prompt += "\n"
    
  prompt += textwrap.dedent("""Below are three patients that didn't survive\n
  """)

    
  for _, negative_patient in negative_examples.iterrows():
    prompt += patient_prompt(negative_patient) + "\n"
  prompt += "\n"
  
  prompt += "Below is the data of the patient of interest\n\n"

  prompt += patient_prompt(patient) + "\n"

  prompt += textwrap.dedent("""\
    Please provide for this patient a probabilty of survival between 0 and 1!
    Respond only with the probability, no text or other explanations!
  """)

  response: ChatResponse = chat(
    model='gemma2:2b', 
    messages=[
      {
        'role': 'user',
        'content': prompt,
      },
    ],
    format={
      "type": "object",
      "properties": {
        "probability": {
          "type": "number",
          "maximum": 1,
          "minimum": 0
        }
      },
      "required": ["probability"],
      "additionalProperties": False
    } 
  )
  
  probability = json.loads(response.message.content)["probability"]
  return probability


df_a = pd.read_parquet(script_dir / '../../data/set-a-imputed.parquet').sort_values(by=['RecordID','Time'])
df_c = pd.read_parquet(script_dir / '../../data/set-c-imputed.parquet').sort_values(by=['RecordID','Time'])

df_a = compute_simple_features(df_a)
df_c = compute_simple_features(df_c)


outcomes_a = pd.read_csv(script_dir / '../../data/Outcomes-a.txt').sort_values('RecordID')[['RecordID', 'In-hospital_death']]
outcomes_a = outcomes_a.set_index('RecordID').iloc[:, 0]

outcomes_c = pd.read_csv(script_dir / '../../data/Outcomes-c.txt')[['RecordID', 'In-hospital_death']]
outcomes_c = outcomes_c.set_index('RecordID').iloc[:, 0]



record_ids_0 = outcomes_a[outcomes_a == 0].index.to_series().sample(n=3, random_state=42).tolist()
record_ids_1 = outcomes_a[outcomes_a == 1].index.to_series().sample(n=3, random_state=42).tolist()


patients_0 = df_a[df_a.index.isin(record_ids_0)]
patients_1 = df_a[df_a.index.isin(record_ids_1)]  


data_dict = {
   "RecordID": [],
   "Probability": [],
   "Prediction": []
}

i = 1
for record_id, patient in df_c.iterrows():
    print(i)
    probability = prompt_llm(patient, patients_0, patients_1)
    data_dict["RecordID"].append(record_id) 
    data_dict["Probability"].append(probability)
    data_dict["Prediction"].append(0 if probability > 0.5 else 1)
    i += 1




llm_df = pd.DataFrame(data_dict)
llm_df.to_parquet(script_dir / f"../../data/foundation-model-predictions.parquet", index=False)






