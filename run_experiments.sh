#!/bin/bash

for MODEL in distilbert roberta; do
    for METHOD in Occlusion Shap LIME; do
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=xai_${MODEL}_${METHOD}
#SBATCH --output=logs/output/${MODEL}_${METHOD}.log
#SBATCH --error=logs/error/${MODEL}_${METHOD}.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=owner1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --mem=24G

echo "Job started: \$(date)"
python run_experiments.py --method ${METHOD} --model ${MODEL}
echo "Job ended: \$(date)"

EOF
    done
done