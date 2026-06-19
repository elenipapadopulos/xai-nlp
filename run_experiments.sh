#!/bin/bash

for MODEL in distilbert roberta roberta-sst2; do
    for METHOD in Occlusion Shap LIME; do
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=xai_${MODEL}_${METHOD}_aux_not
#SBATCH --output=logs/output/${MODEL}_${METHOD}_aux_not.log
#SBATCH --error=logs/error/${MODEL}_${METHOD}_aux_not.log
#SBATCH --nodes=1
#SBATCH --partition=owner1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --mem=24G

mkdir logs/output/${MODEL}_${METHOD}_not_start.log logs/error/${MODEL}_${METHOD}_not_start.log

echo "Job started: \$(date)"
python run_experiments.py --method ${METHOD} --model ${MODEL} --folder not_start
echo "Job ended: \$(date)"

EOF
    done
done