# Set environment variables for OpenMP
export CFLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include"
export LDFLAGS="-L$(brew --prefix libomp)/lib -lomp"

# Install the signatory package
pip install .