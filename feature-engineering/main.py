import json
import pandas as pd

with open('./train.json') as f:
    data = json.load(f)
    dataset = pd.DataFrame(data)

dataset.head()