import subprocess
import math

NODES = 1
TARGET_SECONDS = 15
WALLTIME_MINUTES_PER_JOB = math.ceil(TARGET_SECONDS * 4 / 60)  # Double the target seconds, convert to minutes, round up
WALLTIME_MINUTES = WALLTIME_MINUTES_PER_JOB * 100
WALLTIME_HOURS = math.floor(WALLTIME_MINUTES / 60)
WALLTIME_MINUTES = WALLTIME_MINUTES % 60
SQUARE_WAVE_EXE = "./power_benchmarking/benchmark_step_function/bin/step_function"
SQUARE_WAVE_SRC = "./power_benchmarking/benchmark_step_function/src/main.cpp"

SBATCH_PREAMBLE = f'''#!/usr/bin/env bash
#SBATCH -A gen010
#SBATCH -p batch
#SBATCH -N {NODES}
#SBATCH --exclusive
#SBATCH -t {WALLTIME_HOURS:02}:{WALLTIME_MINUTES:02}:00
#SBATCH -J multi-node-benchmarks
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# -----------------------------------------------------------------------------
# Compilation (run before submitting job)
# -----------------------------------------------------------------------------
# source ../setup-env.sh
# scorep-hipcc -std=c++17 -O3 -w -o ./power_benchmarking/benchmark_step_function/bin/step_function ./power_benchmarking/benchmark_step_function/src/main.cpp -I$MPICH_DIR/include -L$MPICH_DIR/lib -lmpi -DN_ITER=64 --offload-arch=gfx90a -lmpi_gtl_hsa

# export VECTOR_SIZE=1073741824 # Avg 7ms per kernel
export VECTOR_SIZE=134217728 # Avg ~0.9ms per kernel (suitable for 1ms period)

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
export ROCM_VERSION="${{ROCM_VERSION:-6.4.1}}"

module load libfabric/1.22.0
module load perftools-base/24.11.0
module load PrgEnv-amd/8.6.0
module load amd/${{ROCM_VERSION}}
module load cray-mpich/8.1.31
module load rocm/${{ROCM_VERSION}}

mkdir -p results

# Your local hooks
source ../setup-env.sh
source ./setup-run-params.sh

set -euo pipefail

# MPI + GPU (Cray MPICH + ROCm)
export MPICH_GPU_SUPPORT_ENABLED=1

echo "Job $SLURM_JOB_ID started: $(date)"
echo "ROCm PATH: ${{ROCM_PATH:-<unset>}}"
echo "ROCm VERSION: $ROCM_VERSION"
echo "NodeList: $SLURM_NODELIST"

# Resolve nodes in this allocation
mapfile -t NODES < <(scontrol show hostnames "$SLURM_NODELIST")
echo "Using ${{#NODES[@]}} nodes: ${{NODES[*]}}"

for i in "${{!NODES[@]}}"; do
    node="${{NODES[$i]}}"
    echo "Launching square wave on ${{node}} (NODE_NUMBER=${{i}}) with target time {TARGET_SECONDS}s"
'''

SLURM_SUFFIX = '''
done
wait
echo "Job $SLURM_JOB_ID completed: $(date)"
'''
# python3 generate-periods.py --srun-prefix "srun -N1 -w "$node" "${SRUN_BASE[@]}" env NODE_NUMBER="$i" ./wrap-proc.sh ./square-wave" --target-seconds 20 --format commands --use-single-gcd > master-run-square-wave-single-gcd.sbatch
# python3 generate-periods.py --target-seconds 20 --format commands >> master-run-square-wave-both-gcd.sbatch


shell_cmd = f'''python3 generate-periods.py --srun-prefix "srun -N1 -w \"\$node\" env NODE_NUMBER=\"\$i\"" --target-seconds {TARGET_SECONDS} --format commands --use-single-gcd --bench-cmd "./wrap-proc.sh {SQUARE_WAVE_EXE}"'''
# Run shell command to get the slurm script content
shell_output = subprocess.check_output(shell_cmd, shell=True, text=True)

with open("master-run-square-wave-single-gcd.sbatch", "w") as f:
    f.write(SBATCH_PREAMBLE)
    # Indent shell output by 4 spaces
    for line in shell_output.splitlines():
        f.write("    " + line + "\n")
    f.write(SLURM_SUFFIX)
print("SBATCH script 'master-run-square-wave-single-gcd.sbatch' generated.")

shell_cmd = f'''python3 generate-periods.py --srun-prefix "srun -N1 -w \"\$node\" env NODE_NUMBER=\"\$i\"" --target-seconds {TARGET_SECONDS} --format commands --bench-cmd "./wrap-proc.sh {SQUARE_WAVE_EXE}"'''
# Run shell command to get the slurm script content
shell_output = subprocess.check_output(shell_cmd, shell=True, text=True)

with open("master-run-square-wave-batch-both-gcd.sbatch", "w") as f:
    f.write(SBATCH_PREAMBLE)
    # Indent shell output by 4 spaces
    for line in shell_output.splitlines():
        f.write("    " + line + "\n")
    f.write(SLURM_SUFFIX)
print("SBATCH script 'master-run-square-wave-batch-both-gcd.sbatch' generated.")