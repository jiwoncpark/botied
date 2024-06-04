import random
import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning)
from botied import acquisitions
from botorch.utils.transforms import unnormalize, normalize
from botied.metrics.cdf_indicator import CDFIndicator
from omegaconf import OmegaConf
from hydra.utils import instantiate
import wandb
import logging
logger = logging.getLogger(__name__)


class ExperimentManager:
    def __init__(
            self, objective, model_wrapper, acquisitions_kwargs,
            cdf_indicator: CDFIndicator, dtype, device):
        objective.problem = objective.problem.to(device, dtype)
        self.objective = objective
        self.model_wrapper = model_wrapper
        self.dtype = dtype
        self.device = device
        self.instantiate_all_acquisitions(**acquisitions_kwargs)
        self.cdf_obj = cdf_indicator

    def instantiate_all_acquisitions(self, **kwargs):
        self.qnparego = acquisitions.qNParEGO()
        self.botied_copula_cdf = acquisitions.BOtiedCDF(
            cdf_model=instantiate(kwargs['botied_copula_cdf']['cdf_model'])
        )
        for cdf_type in ['copula', 'copula_inverse', 'mvn', 'empirical', 'kde']:
            # Fit CDF on all samples, mean-aggregate CDF scores (v1)
            setattr(
                self, f'botied_{cdf_type}_cdf', acquisitions.BOtiedCDF(
                    cdf_model=instantiate(
                        kwargs[f'botied_{cdf_type}_cdf']['cdf_model']),
                    apply_ref_point=kwargs[f'botied_{cdf_type}_cdf']['apply_ref_point'],
                )
            )
            logger.info(f"CDF type: {cdf_type}, apply_ref_point: {kwargs[f'botied_{cdf_type}_cdf']['apply_ref_point']}")
            # Fit CDF on posterior means (v2)
            setattr(
                self, f'botied_{cdf_type}_cdf_of_means',
                acquisitions.BOtiedCDF(
                    cdf_model=instantiate(
                        kwargs[f'botied_{cdf_type}_cdf']['cdf_model']),
                    aggregation='cdf_of_means',
                    apply_ref_point=kwargs[f'botied_{cdf_type}_cdf']['apply_ref_point'],
                )
            )
            # Fit CDF on posterior means, eval on samples, mean-aggregate CDF scores (v3)
            setattr(
                self, f'botied_{cdf_type}_cdf_of_means_eval_samples',
                acquisitions.BOtiedCDF(
                    cdf_model=instantiate(
                        kwargs[f'botied_{cdf_type}_cdf']['cdf_model']),
                    aggregation='cdf_of_means_eval_samples',
                    apply_ref_point=kwargs[f'botied_{cdf_type}_cdf']['apply_ref_point'],
                )
            )
        self.qnehvi = acquisitions.qNEHVI(**kwargs['qnehvi'])
        self.pes = acquisitions.PES(**kwargs['pes'])
        self.mes = acquisitions.MES(**kwargs['mes'])
        self.jes = acquisitions.JES(**kwargs['jes'])
        self.random = acquisitions.Random(**kwargs['random'])

    def select(self, acq_name: str, **kwargs):
        """
        To be deprecated, for both subset selection and optimization
        """
        select_fn = getattr(acquisitions, f'select_{acq_name}')
        return select_fn(**kwargs)

    def _update_metrics(self, data, existing_data_dict):
        """

        Parameters
        ----------
        data : dict
            Data to add (e.g., with keys `x`, `y`, `clean_y`)
        existing_data_dict : dict
            Data dict to update with the new `data`. Must have matching keys
            with `data` with list-type values.

        """
        for k, v in data.items():
            if existing_data_dict.get(k) is None:
                existing_data_dict[k] = []
            existing_data_dict[k].append(v)
        return existing_data_dict

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def select_from_random_pool(self, **kwargs):
        self.seed_everything(kwargs['seed'])
        # Define splits into train, rounds, test
        self.objective.set_split(kwargs['n_batches'])
        kwargs = OmegaConf.structured(kwargs)
        # Generate initial data, populate each data dict
        init_data = self.objective.get_initial(n=kwargs['initial_n'])
        # Populate data with init_data for each acq function
        data = {}
        for acq_name in kwargs.acq_list:
            data[acq_name] = []
            data[acq_name].append(init_data)
        # Compute initial hypervolume
        ref_point = self.objective.problem.ref_point
        logger.info(f"ref point: {ref_point}")
        bd = DominatedPartitioning(
            ref_point=ref_point,
            Y=init_data['clean_y']  # observed (noisy) y
        )
        # Compute initial CDF scores
        init_cdf = self.cdf_obj.compute_copula_cdf(init_data['clean_y'])
        init_metrics = {
            'seed': kwargs['seed'],
            'volume': bd.compute_hypervolume().item(),
            'copula_cdf': init_cdf}
        # Populate logging dict with initial metrics for each acq function
        metrics = {}
        for acq_name in kwargs.acq_list:
            metrics[acq_name] = {}
            metrics[acq_name] = self._update_metrics(
                init_metrics, metrics[acq_name])
        wandb.log(init_metrics, step=0)

        # Run BO rounds
        for round_idx in range(1, kwargs.n_batches + 1):
            # Sample random candidate pool (shared by all acquisitions)
            pool = self.objective.get_pool(
                n=kwargs.batch_size*kwargs.select_factor,
                round_idx=round_idx)
            for acq_name in kwargs.acq_list:
                context = self.objective.concat(data[acq_name])
                # Fit model on normalized x
                model, likelihood = self.model_wrapper.fit_model(
                    train_x=context['scaled_x'],
                    # train_x=normalize(context['x'], self.objective.problem.bounds),
                    train_obj=context['scaled_y'],
                    bounds=self.objective.problem.bounds)
                # Get predictive samples for candidate pool
                # TODO: make nicer
                n_train = context['y'].shape[0] if isinstance(
                    context, dict) else len(context)
                model_input = self.objective.concat_input(
                    data[acq_name] + [pool]
                )
                pred_sample = self.model_wrapper.sample_predictions(
                    model=model,
                    likelihood=likelihood,
                    input_x=model_input['scaled_x'],
                    # input_x=normalize(model_input['x'], self.objective.problem.bounds),
                    n_samples=kwargs.model.n_pred_samples)
                # Select subset
                results = getattr(self, acq_name).select_subset(
                    f_cand=pred_sample[:, n_train:, :],
                    f_baseline=pred_sample[:, :n_train, :],
                    q_batch_size=kwargs.batch_size,
                    ref_point=ref_point,
                    model=model,
                    x_cand=model_input['scaled_x'][n_train:],
                    # x_cand=normalize(
                    #     model_input['x'][n_train:],
                    #     self.objective.problem.bounds),
                    x_baseline=model_input['scaled_x'][:n_train],
                    # x_baseline=normalize(
                    #     model_input['x'][:n_train],
                    #     self.objective.problem.bounds)
                )
                # Slice selected from pool, append to training set
                to_append = self.objective.slice(pool, results['selected_idx'])
                data[acq_name].append(to_append)
                # Update progress
                bd = DominatedPartitioning(
                    ref_point=ref_point,
                    Y=self.objective.concat(data[acq_name], collate=True)['clean_y'])
                new_cdf = self.cdf_obj.compute_copula_cdf(to_append['clean_y'])
                round_metrics = {
                    f'volume/{acq_name}': bd.compute_hypervolume().item(),
                    f'copula_cdf/{acq_name}': new_cdf
                }
                # Store in running dict and also log in wandb
                metrics[acq_name] = self._update_metrics(
                    round_metrics, metrics[acq_name])
                wandb.log(round_metrics, step=round_idx)
                seed = str(kwargs['seed'])
                torch.save(metrics[acq_name], f'{acq_name}_metrics_{seed}.pt')
                torch.save(data, f'data_{seed}.pt')
        self.log_lookback_cdf_scores(kwargs.acq_list, data, kwargs.n_batches)

    def optimize_over_design_space(self, **kwargs):
        self.seed_everything(kwargs['seed'])
        # Define splits into train, rounds, test
        self.objective.set_split(kwargs['n_batches'])
        kwargs = OmegaConf.structured(kwargs)
        # Generate initial data, populate each data dict
        init_data = self.objective.get_initial(n=kwargs['initial_n'])
        # Populate data with init_data for each acq function
        data = {}
        for acq_name in kwargs.acq_list:
            data[acq_name] = []
            data[acq_name].append(init_data)
        # Compute initial hypervolume
        ref_point = self.objective.problem.ref_point
        logger.info(f"ref point: {ref_point}")
        bd = DominatedPartitioning(
            ref_point=ref_point,
            Y=init_data['clean_y']  # observed (noisy) y
        )
        # Compute initial CDF scores
        init_cdf = self.cdf_obj.compute_copula_cdf(init_data['clean_y'])
        init_metrics = {
            'seed': kwargs['seed'],
            'volume': bd.compute_hypervolume().item(),
            'copula_cdf': init_cdf}
        # Populate logging dict with initial metrics for each acq function
        metrics = {}
        for acq_name in kwargs.acq_list:
            metrics[acq_name] = {}
            metrics[acq_name] = self._update_metrics(
                init_metrics, metrics[acq_name])
        wandb.log(init_metrics, step=0)

        # Run BO rounds
        logger.info("Entering BO loop")
        for round_idx in range(1, kwargs.n_batches + 1):
            logger.info(f"Round: {round_idx}")
            for acq_name in kwargs.acq_list:
                logger.info(f"Acquisition: {acq_name}")
                # Init models cold, fit on data so far
                context = self.objective.concat(data[acq_name])
                model, _ = self.model_wrapper.fit_model(
                    train_x=context['scaled_x'],
                    # train_x=normalize(context['x'], self.objective.problem.bounds),
                    train_obj=context['scaled_y'],
                    bounds=self.objective.problem.bounds)
                # Find optimal candidates
                results = getattr(self, acq_name).optimize(
                    model=model,
                    train_x=context['scaled_x'],
                    # train_x=normalize(context['x'], self.objective.problem.bounds),
                    bounds=self.objective.problem.bounds,
                    ref_point=ref_point,
                    q_batch_size=kwargs.batch_size,
                )
                to_append = self.objective.evaluate(results['new_x'])
                data[acq_name].append(to_append)
                # Update progress
                acquired_so_far = self.objective.concat(
                    data[acq_name], collate=True)['clean_y']
                bd = DominatedPartitioning(
                    ref_point=ref_point,
                    Y=acquired_so_far)
                new_cdf = self.cdf_obj.compute_copula_cdf(to_append['clean_y'])
                round_metrics = {
                    f'volume/{acq_name}': bd.compute_hypervolume().item(),
                    f'copula_cdf/{acq_name}': new_cdf
                }
                # Store in running dict and also log in wandb
                metrics[acq_name] = self._update_metrics(
                    round_metrics, metrics[acq_name])
                wandb.log(round_metrics, step=round_idx)
                if kwargs['show_scatterplot']:
                    table = wandb.Table(
                        data=acquired_so_far.cpu().numpy()[:, [0, 1]],
                        columns=["Obj 1", "Obj 2"])
                    wandb.log(
                        {f"scatterplot/{acq_name}": wandb.plot.scatter(
                            table, "Obj 1", "Obj 2",
                            title=acq_name)},
                        step=round_idx)
                seed = str(kwargs['seed'])
                torch.save(metrics[acq_name], f'{acq_name}_metrics_{seed}.pt')
                torch.save(data, f'data_{seed}.pt')
        self.log_lookback_cdf_scores(kwargs.acq_list, data, kwargs.n_batches)

    def log_lookback_cdf_scores(
            self, acq_list: list, data: dict, n_batches: int):
        logger.info("Logging lookback CDF scores")
        wandb.define_metric("lookback_step")
        evaluated_points = {}
        # Collate data across iterations
        for acq_name in acq_list:
            evaluated_points[acq_name] = self.objective.concat(
                data[acq_name], collate=False, to_numpy=True)['clean_y']
        # Also collate data across acq functions and compute lookback CDF
        lookback_cdf = self.cdf_obj.compute_lookback_copula_cdf(
            evaluated_points)
        # Log
        for acq_name in acq_list:
            wandb.define_metric(
                f"lookback_copula_cdf/{acq_name}", step_metric="lookback_step")
            for iter_i in range(1, n_batches + 1):
                wandb.log(
                    {f"lookback_copula_cdf/{acq_name}": lookback_cdf[acq_name][iter_i]},
                )
