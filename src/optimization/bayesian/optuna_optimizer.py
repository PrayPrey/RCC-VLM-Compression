"""
Bayesian optimization using Optuna for hyperparameter search.

This module implements Bayesian optimization for finding optimal
compression parameters across the cascade pipeline.
"""

import optuna
from optuna import Trial, Study
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any, Callable, Union
import numpy as np
import logging
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for Bayesian optimization."""
    study_name: str = "rcc_compression"
    direction: str = "maximize"  # maximize or minimize
    n_trials: int = 100
    timeout: Optional[float] = None  # Timeout in seconds
    n_jobs: int = 1  # Parallel trials
    sampler: str = "tpe"  # tpe, cmaes, random
    pruner: str = "median"  # median, hyperband, none
    storage: Optional[str] = None  # Database URL for distributed optimization
    load_if_exists: bool = True
    seed: int = 42


class BayesianOptimizer:
    """Bayesian optimization for compression hyperparameters."""

    def __init__(self, config: OptimizationConfig):
        """
        Initialize Bayesian optimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.study = None
        self.best_params = None
        self.best_value = None

        # Setup sampler
        if config.sampler == "tpe":
            self.sampler = TPESampler(seed=config.seed)
        elif config.sampler == "cmaes":
            self.sampler = CmaEsSampler(seed=config.seed)
        else:
            self.sampler = None

        # Setup pruner
        if config.pruner == "median":
            self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif config.pruner == "hyperband":
            self.pruner = HyperbandPruner()
        else:
            self.pruner = None

    def create_study(self) -> Study:
        """
        Create or load Optuna study.

        Returns:
            Optuna study instance
        """
        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.config.storage,
            load_if_exists=self.config.load_if_exists
        )

        logger.info(f"Created study: {self.config.study_name}")
        return self.study

    def optimize(self,
                objective: Callable[[Trial], float],
                callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Run optimization.

        Args:
            objective: Objective function
            callbacks: Optional callbacks

        Returns:
            Best parameters and results
        """
        if self.study is None:
            self.create_study()

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            callbacks=callbacks
        )

        # Get best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        results = {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'best_trial': self.study.best_trial.number
        }

        logger.info(f"Optimization complete. Best value: {self.best_value:.4f}")
        logger.info(f"Best params: {self.best_params}")

        return results

    def suggest_compression_params(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest compression parameters for trial.

        Args:
            trial: Optuna trial

        Returns:
            Suggested parameters
        """
        params = {}

        # DARE parameters
        params['dare_sparsity'] = trial.suggest_float('dare_sparsity', 0.7, 0.95)
        params['dare_schedule'] = trial.suggest_categorical(
            'dare_schedule', ['linear', 'cosine', 'exponential']
        )
        params['dare_iterations'] = trial.suggest_int('dare_iterations', 5, 20)

        # Nullu parameters
        params['nullu_rank_ratio'] = trial.suggest_float('nullu_rank_ratio', 0.3, 0.7)
        params['nullu_energy_threshold'] = trial.suggest_float(
            'nullu_energy_threshold', 0.9, 0.99
        )
        params['nullu_rank_method'] = trial.suggest_categorical(
            'nullu_rank_method', ['energy', 'adaptive', 'hybrid']
        )

        # AlphaEdit parameters
        params['alpha_lr'] = trial.suggest_float('alpha_lr', 1e-5, 1e-2, log=True)
        params['alpha_epochs'] = trial.suggest_int('alpha_epochs', 5, 20)
        params['alpha_importance'] = trial.suggest_categorical(
            'alpha_importance', ['gradient', 'magnitude', 'taylor', 'hybrid']
        )

        # Training parameters
        params['kd_temperature'] = trial.suggest_float('kd_temperature', 1.0, 10.0)
        params['kd_weight'] = trial.suggest_float('kd_weight', 0.1, 0.9)

        return params

    def suggest_training_params(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest training hyperparameters.

        Args:
            trial: Optuna trial

        Returns:
            Suggested training parameters
        """
        params = {}

        # Optimizer
        params['optimizer'] = trial.suggest_categorical(
            'optimizer', ['adam', 'adamw', 'sgd']
        )
        params['learning_rate'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        params['weight_decay'] = trial.suggest_float('weight_decay', 0, 0.1)

        # Scheduler
        params['scheduler'] = trial.suggest_categorical(
            'scheduler', ['cosine', 'step', 'exponential', 'none']
        )

        if params['scheduler'] == 'step':
            params['step_size'] = trial.suggest_int('step_size', 5, 50)
            params['gamma'] = trial.suggest_float('gamma', 0.1, 0.9)

        # Batch size
        params['batch_size'] = trial.suggest_categorical(
            'batch_size', [16, 32, 64, 128, 256]
        )

        # Gradient clipping
        params['gradient_clip'] = trial.suggest_float('gradient_clip', 0.1, 10.0)

        return params

    def create_compression_objective(self,
                                    model_fn: Callable,
                                    train_fn: Callable,
                                    eval_fn: Callable) -> Callable:
        """
        Create objective function for compression optimization.

        Args:
            model_fn: Function to create model
            train_fn: Function to train model
            eval_fn: Function to evaluate model

        Returns:
            Objective function
        """
        def objective(trial: Trial) -> float:
            # Suggest parameters
            compression_params = self.suggest_compression_params(trial)
            training_params = self.suggest_training_params(trial)

            # Combine parameters
            params = {**compression_params, **training_params}

            try:
                # Create and compress model
                model = model_fn(params)

                # Train compressed model
                train_metrics = train_fn(model, params, trial)

                # Report intermediate values for pruning
                if self.config.pruner and hasattr(trial, 'report'):
                    for epoch, metric in enumerate(train_metrics.get('val_accuracy', [])):
                        trial.report(metric, epoch)

                        # Check if trial should be pruned
                        if trial.should_prune():
                            raise optuna.TrialPruned()

                # Final evaluation
                eval_metrics = eval_fn(model, params)

                # Calculate objective value
                compression_ratio = eval_metrics.get('compression_ratio', 1.0)
                accuracy = eval_metrics.get('accuracy', 0.0)

                # Combined objective (maximize compression while maintaining accuracy)
                objective_value = accuracy * np.log(compression_ratio + 1)

                return objective_value

            except Exception as e:
                logger.error(f"Trial failed: {e}")
                return float('-inf') if self.config.direction == 'maximize' else float('inf')

        return objective

    def get_pareto_front(self,
                        objectives: List[str] = ['accuracy', 'compression_ratio']) -> List[Dict]:
        """
        Get Pareto front from completed trials.

        Args:
            objectives: List of objective names

        Returns:
            List of Pareto optimal trials
        """
        if self.study is None:
            return []

        # Extract objective values
        trials_data = []
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_dict = {
                    'number': trial.number,
                    'params': trial.params,
                    'values': {}
                }

                for obj in objectives:
                    if obj in trial.user_attrs:
                        trial_dict['values'][obj] = trial.user_attrs[obj]

                if len(trial_dict['values']) == len(objectives):
                    trials_data.append(trial_dict)

        # Find Pareto front
        pareto_front = []
        for trial in trials_data:
            is_pareto = True
            for other in trials_data:
                if trial['number'] != other['number']:
                    # Check if other dominates trial
                    dominates = all(
                        other['values'][obj] >= trial['values'][obj]
                        for obj in objectives
                    ) and any(
                        other['values'][obj] > trial['values'][obj]
                        for obj in objectives
                    )

                    if dominates:
                        is_pareto = False
                        break

            if is_pareto:
                pareto_front.append(trial)

        return pareto_front

    def visualize_optimization(self,
                              save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create visualization of optimization results.

        Args:
            save_path: Path to save visualizations

        Returns:
            Dictionary of visualization data
        """
        if self.study is None:
            return {}

        viz_data = {}

        # Optimization history
        viz_data['history'] = [
            {
                'trial': t.number,
                'value': t.value,
                'params': t.params
            }
            for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]

        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            viz_data['param_importance'] = importance
        except:
            viz_data['param_importance'] = {}

        # Parameter distributions
        viz_data['param_distributions'] = {}
        for param_name in self.study.best_params.keys():
            values = [
                t.params.get(param_name)
                for t in self.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE and param_name in t.params
            ]
            if values:
                viz_data['param_distributions'][param_name] = {
                    'values': values,
                    'mean': np.mean(values) if all(isinstance(v, (int, float)) for v in values) else None,
                    'std': np.std(values) if all(isinstance(v, (int, float)) for v in values) else None
                }

        if save_path:
            # Save visualization data
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w') as f:
                json.dump(viz_data, f, indent=2, default=str)

            # Create plots if optuna visualization is available
            try:
                import optuna.visualization as vis

                # Optimization history
                fig = vis.plot_optimization_history(self.study)
                fig.write_html(str(save_path.parent / "optimization_history.html"))

                # Parameter importance
                fig = vis.plot_param_importances(self.study)
                fig.write_html(str(save_path.parent / "param_importance.html"))

                # Parallel coordinate plot
                fig = vis.plot_parallel_coordinate(self.study)
                fig.write_html(str(save_path.parent / "parallel_coordinate.html"))

            except ImportError:
                logger.warning("Optuna visualization not available")

        return viz_data

    def save_study(self, path: str):
        """
        Save study to file.

        Args:
            path: Save path
        """
        if self.study is None:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save with pickle
        with open(path, 'wb') as f:
            pickle.dump(self.study, f)

        # Also save best params as JSON
        json_path = path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_value': self.best_value,
                'n_trials': len(self.study.trials)
            }, f, indent=2)

        logger.info(f"Saved study to {path}")

    def load_study(self, path: str):
        """
        Load study from file.

        Args:
            path: Load path
        """
        path = Path(path)

        if path.exists():
            with open(path, 'rb') as f:
                self.study = pickle.load(f)

            self.best_params = self.study.best_params
            self.best_value = self.study.best_value

            logger.info(f"Loaded study from {path}")
        else:
            logger.warning(f"Study file not found: {path}")