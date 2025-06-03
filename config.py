# Paths to data files
TENSOR_PATH = "data/PRJEB53403/genefamilies/after_intersection/tensor.npy"
SAMPLE_LIST_PATH = "data/PRJEB53403/genefamilies/after_intersection/sample_list.npy"
BACTERIA_LIST_PATH = "data/PRJEB53403/genefamilies/after_intersection/bacteria_list.npy"
GENE_LIST_PATH = "data/PRJEB53403/genefamilies/after_intersection/gene_families_list.npy"

# Output directory for processed data
NAME = "26/05/2025"
SERIAL_NUM = 8
EVAL_OUTPUT_DIR = f"eval_data/Run_{SERIAL_NUM}"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
EMBEDDING_DIM = 128
λ1 = 6.72883E-09
λ2 = 0.788196774
λ3 = 0.211803219	
LAMBDA_WEIGHT = [λ1, λ2, λ3] 




