#!/bin/bash

for METHOD in Occlusion Shap LIME; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=xai_${METHOD}
#SBATCH --output=logs/output/${METHOD}.log
#SBATCH --error=logs/error/${METHOD}.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=owner1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --mem=24G

# python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoTokenizer.from_pretrained('siebert/sentiment-roberta-large-english'); AutoModelForSequenceClassification.from_pretrained('siebert/sentiment-roberta-large-english')"

python run_experiments.py --method ${METHOD} --model roberta
EOF
done