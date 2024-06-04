# BOtied: Multi-objective Bayesian optimization with tied multivariate ranks

## Installation

Clone this repo and cd into the root. Then run
```
source setup.sh

```

(Optional) Create a kernel with the corresponding environment
```
pip install --user ipykernel
python -m ipykernel install --user --name botied --display-name "Python (botied)"
```

## Running the experiments (on the LSF system)

To run sequential optimization (where the acquisition function is optimized over the design space every iteration) with access to an evaluable test function, use the following example.
```
source scripts/jobs/submit_penicillin_jobs.sh
```

To run batched selection simulating multi-objective Bayesian optimization given an annotated dataset, use the following example.
```
source scripts/jobs/submit_caco2_plus_jobs.sh
```

## Citation

If you use the code, please cite the following paper
```
@inproceedings{park2024botied,
  title={BOtied: Multi-objective Bayesian optimization with tied multivariate ranks},
  author={Park, Ji Won and Tagasovska, Nata{\v{s}}a and Maser, Michael and Ra, Stephen and Cho, Kyunghyun},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```
