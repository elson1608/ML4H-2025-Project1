import pandas as pd
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from pathlib import Path

script_dir = Path(__file__).parent

for set_type  in ['a', 'b', 'c']:
    df = pd.read_parquet(script_dir / f'../../data/set-{set_type}-imputed-scaled.parquet') \
        .sort_values(by=['RecordID','Time'])
    
    df['Time'] = df['Time'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60)

    print(f'Extracting features for set {set_type}...')

    # Extract features with a lower chunksize to avoid memory overflow    
    df_tsfresh = extract_features(
        df, 
        column_id='RecordID', 
        column_sort='Time',
        n_jobs=0,
        default_fc_parameters = EfficientFCParameters()      
    )

    df_tsfresh.to_parquet(script_dir / f'../../data/tsfresh-set-{set_type}.parquet')