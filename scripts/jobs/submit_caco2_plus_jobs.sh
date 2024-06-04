dataset=caco2_plus
version=10

# q=4
batch_size=4
echo "icml_${dataset}_q${batch_size}_v${version}"
for seed in {0..30};
do bsub -n 4 -We 180 -q "short" -R "span[hosts=1] rusage[mem=100GB]" python scripts/run_selection_experiments.py -m +multirun=${dataset}_no_es ++wandb.project=icml_${dataset}_q${batch_size}_v${version} run_kwargs.seed=$seed run_kwargs.batch_size=4 run_kwargs.n_batches=20
done


# gpu (query gpu for ES!)

# # q=1
# for seed in {0..19};
# # do echo $seed
# do bsub -gpu num=1 -n 1 -R "rusage[mem=100GB] span[ptile=12]" -W 7200 python scripts/run_selection_experiments.py -m +multirun=caco2_plus ++wandb.project=icml_caco2_plus_q1_v2 run_kwargs.seed=$seed run_kwargs.batch_size=1 run_kwargs.n_batches=80
# done

# # q=2
# for seed in {0..19};
# # do echo $seed
# do bsub -gpu num=1 -n 1 -R "rusage[mem=100GB] span[ptile=12]" -W 7200 python scripts/run_selection_experiments.py -m +multirun=caco2_plus ++wandb.project=icml_caco2_plus_q2_v2 run_kwargs.seed=$seed run_kwargs.batch_size=2 run_kwargs.n_batches=40
# done

# # q=4
# for seed in {0..19};
# # do echo $seed
# do bsub -gpu num=1 -n 1 -R "rusage[mem=100GB] span[ptile=12]" -W 7200 python scripts/run_selection_experiments.py -m +multirun=caco2_plus ++wandb.project=icml_caco2_plus_q4_v2 run_kwargs.seed=$seed run_kwargs.batch_size=4 run_kwargs.n_batches=20
# done


