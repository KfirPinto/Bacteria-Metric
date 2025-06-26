import numpy as np
import pandas as pd

data = np.load("test_bacteria.npy", allow_pickle=True)
df = pd.DataFrame(data)
df.to_csv("test_bacteria.csv", index=False)
