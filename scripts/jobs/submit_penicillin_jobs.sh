dataset=penicillin
version=19
# Change num_seeds

# q=1
batch_size=1
echo "icml_${dataset}_q${batch_size}_v${version}"
for seed in {0..19};
do
    bsub -n 4 -We 600 -q "long" -R "span[hosts=1] rusage[mem=50GB]" -o "scripts/logs/%J.out" python scripts/run_optimization_experiments.py -m +multirun=${dataset}_no_es ++wandb.project=icml_${dataset}_q${batch_size}_v${version}_scratch run_kwargs.seed=$seed run_kwargs.batch_size=$batch_size run_kwargs.n_batches=80
    sleep 5
done
