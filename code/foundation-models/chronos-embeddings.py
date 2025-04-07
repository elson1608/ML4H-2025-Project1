from pathlib import Path
import pandas as pd
import torch
from chronos import ChronosPipeline


script_dir = Path(__file__).parent

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map="cuda",
    torch_dtype=torch.float32,
)


def get_embeddings(df):



    stacked_tensor = torch.stack(
        [torch.tensor(group.drop(columns=['RecordID']).values, dtype=torch.float32) 
        for _, group in df.groupby('RecordID')]
    )

    embeddings = []


    for i in range(stacked_tensor.size(-1)):
        context = stacked_tensor[:, :, i]


        # Compute the embedding for this column (feature)
        embedding, _ = pipeline.embed(context)

        # We get an embeding for each timestep so we average over all timesteps
        embedding = embedding.mean(axis=1)

        embeddings.append(embedding)
        
        print(f"feature {i} done")


    # Stack embeddings along the last dimension (feature dimension)
    X_train = torch.stack(embeddings, dim=1)  


    return X_train   





for set_type in ['a', 'b', 'c']:
    df = pd.read_parquet(script_dir / f'../../data/set-{set_type}-imputed-scaled.parquet').sort_values(by=['RecordID','Time'])
    df = df.drop(columns=['Time'])
    print(f"starting with set-{set_type}")
    embeddings = get_embeddings(df)
    torch.save(embeddings, script_dir / f"../../data/set-{set_type}-chronos-embeddings.pt")  # Save as a .pt file





