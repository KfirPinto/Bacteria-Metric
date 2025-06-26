import numpy as np
import pandas as pd

# your_file.npy contains: [[1, 2, 3], [4, 5, 6]]
data = np.load("test_bacteria.npy", allow_pickle=True)
df = pd.DataFrame(data)
df.to_csv("test_bacteria.csv", index=False)
