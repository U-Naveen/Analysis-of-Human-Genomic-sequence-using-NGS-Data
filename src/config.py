

# Dataset path
DATA_PATH = "data/raw/Human Data Sequnence.txt"

# Train test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training parameters
EPOCHS = 100
BATCH_SIZE = 32

# Model parameters
VOCAB_SIZE = 20000    
EMBEDDING_DIM = 8
NUM_CLASSES = 7

# Output directories
MODEL_PATH = "outputs/models/genomics_cnn_model.h5"
PLOTS_DIR = "outputs/plots/"