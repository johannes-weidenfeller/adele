# Source this file before building the CUDA extensions and before every run:
#     source env.sh
# It assumes the conda environment created in the README (CUDA toolkit installed
# into the env). Adjust CONDA_ENV if you named it differently.
CONDA_ENV=${CONDA_ENV:-adele}
if [ -z "$CONDA_PREFIX" ] || [ "$(basename "$CONDA_PREFIX")" != "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
fi
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
# The nvidia conda packages keep headers/libs under targets/x86_64-linux; torch's
# JIT builds (nerfacc, nvdiffrast, renderutils) look in $CUDA_HOME/include, so expose them.
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# Architectures for the JIT extensions; trim to your GPU (e.g. "8.9" for an RTX 4090) to build faster.
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;8.6;8.9;9.0"}
# Keep the JIT cache per environment (a shared ~/.cache/torch_extensions breaks across envs).
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions_$CONDA_ENV}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
