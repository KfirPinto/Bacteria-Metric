import numpy as np
def checkDim(data):
    pathway_data = np.load(data, allow_pickle=True)
    print(f"Loaded data shape: {pathway_data.shape}")

if __name__ == "__main__":
    
    checkDim("/home/bcrlab/barsapi1/metric/Bacteria-Metric/data/PRJEB53403_168_samples/genefamilies/before_intersection/bacteria_list.npy")
    checkDim("/home/bcrlab/barsapi1/metric/Bacteria-Metric/data/PRJEB53403_168_samples/genefamilies/after_intersection/bacteria_list.npy")
