import pandas as pd
import numpy as np

raw_data = pd.read_csv('testing_scripts/module_dependencies/Below_6.0_SAO.csv').to_numpy()
print(raw_data.shape)
print(raw_data)