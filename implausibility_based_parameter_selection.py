from jax import config

config.update("jax_enable_x64", True)

import copy
from dataclasses import dataclass
import pickle

from beartype.typing import (
    Callable,
    Dict,
    Mapping,
    Tuple,
)
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float
from joblib import (
    Parallel,
    delayed,
)
import numpy as onp  # For Lorenz96
import optax as ox

import gpjax as gpx
from gpjax.dataset import Dataset
from gpjax.decision_making.posterior_handler import PosteriorHandler
from gpjax.decision_making.search_space import ContinuousSearchSpace
from gpjax.gps import AbstractPosterior
from gpjax.typing import (
    Array,
    Float,
    KeyArray,
)
from lorenz96_data_processor import Lorenz96DataProcessor

# Run parameters
SAVE_DIR = "plausibility_trainable_prior_mean/data/"
RANDOM_SEED = 42
INITIAL_NUM_SAMPLES = 40
BATCH_SIZE = 40
TRAINABLE_PRIOR_MEAN = True
PC_PREFIX = "PC"
WAVE_PREFIX = "WAVE"
NUM_WAVES = (
    5  # Number of waves to run *after* initial wave consisting of random samples.
)
THRESHOLD_PLAUSIBILITY = 3.0


def efficient_predict(
    posterior: AbstractPosterior, test_inputs: Float[Array, "N D"], train_data: Dataset
) -> Tuple[Float[Array, "N 1"], Float[Array, "N 1"]]:
    """
    Efficiently predict the posterior mean and variance for a set of test inputs.
    """

    def _predict(x):
        fx = posterior(test_inputs=x, train_data=train_data)
        return fx.mean().squeeze(), fx.covariance().squeeze()

    mean, variance = vmap(_predict)(test_inputs[:, None])
    mean = mean[..., None]
    variance = variance[..., None]
    return mean, variance


@dataclass
class ThresholdPlausibility:
    """
    Utility function builder for the thresholded plausibility acquisition function
    introduced by Lguensat et al.
    (https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2022MS003367). With this,
    we use the surrogate models for each of the principle components to determine
    whether a given parameter configuration is plausible. A parameter configuration is
    deemed to be plausible if the posterior mean of each of the principle components is
    predicted to be within `threshold` standard deviations of the true PCA-transformed metric value.

    Attributes:
        transformed_true_metrics: A dictionary mapping from principle component tag to
        the true metric value for that principle component. These values are obtained by
        running the Lorenz 96 simulation with the true model parameters, and then
        transorming them via PCA/standardisation.
        threshold: Number of standard deviations around the predicted posterior mean
        within which a parameter configuration is deemed to be plausible.
    """

    transformed_true_metrics: Dict[str, Float]
    threshold: float

    def get_plausible_points(
        self,
        x: Float[Array, "N D"],
        wave_posteriors: Mapping[str, Mapping[str, AbstractPosterior]],
        wave_datasets: Mapping[str, Mapping[str, Dataset]],
    ) -> Float[Array, "num_plausible_points D"]:
        """
        Get plausible points from a set of parameter configurations.

        Args:
            wave_posteriors: A dictionary mapping from wave tag to a dictionary mapping
            from principle component tag to the surrogate model posterior for that
            principle component.
            wave_datasets: A dictionary mapping from wave tag to a dictionary mapping
            from principle component tag to the dataset for that principle component.
            key: A JAX PRNG key.

        Returns:
            Points deemed to be plausible.
        """
        if wave_datasets.keys() != wave_posteriors.keys():
            raise ValueError(
                "Wave datasets and wave posteriors dictionaries must have the same keys."
            )
        else:
            plausible_x = x
            # Iterate over waves - a point is only considered to be plausible if it is
            # deemed to be plausible by the models for each wave.
            for wave_tag, wave_posterior_dict in wave_posteriors.items():
                if wave_tag not in wave_datasets.keys():
                    raise ValueError(
                        f"Datasets corresponding to {wave_tag} not found in datasets."
                    )
                else:
                    wave_dataset_dict = wave_datasets[wave_tag]
                    # Iterate over principle components - a point is only considered to
                    # be feasible if it is deemed to be feasible by the models for each
                    # principle component.
                    for (
                        principle_component_tag,
                        principle_component_posterior,
                    ) in wave_posterior_dict.items():
                        if principle_component_tag not in wave_dataset_dict.keys():
                            raise ValueError(
                                f"Dataset corresponding to posterior {principle_component_tag} not found in wave {wave_tag}."
                            )
                        posterior_mean, posterior_var = efficient_predict(
                            posterior=principle_component_posterior,
                            test_inputs=plausible_x,
                            train_data=wave_dataset_dict[principle_component_tag],
                        )
                        true_metric = self.transformed_true_metrics[
                            principle_component_tag
                        ]
                        metric_plausibility = jnp.abs(
                            true_metric - posterior_mean
                        ) / jnp.sqrt(posterior_var)
                        thresholded_metric_plausibility = jnp.where(
                            metric_plausibility < self.threshold, True, False
                        )
                        thresholded_metric_plausibility = jnp.squeeze(
                            thresholded_metric_plausibility
                        )
                        plausible_indices = jnp.where(thresholded_metric_plausibility)[
                            0
                        ]
                        if plausible_indices.shape[0] == 0:
                            # No feasible points
                            return jnp.array([])
                        plausible_x = plausible_x[plausible_indices]

            return plausible_x


@dataclass
class ClimateModelDecisionMaker:
    """
    Dataclass for handling the decision making process for climate model tuning using
    the thresholded plausibility acquisition function introduced by Lguensat et al.
    (https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2022MS003367). With this,
    the decision making process runs for a number of waves. In each wave, we choose a
    batch of sets of parameter configurations to run the climate model with, then run it
    with each of these parameter configurations. This yields a set of metrics for each
    parameter configuration, the number of which we reduce using PCA. We then fit
    surrogate models to each of the principle components of the transformed metrics, and
    use these to decide which parameter configurations to run in the next wave with the
    `ThresholdPlausibilityUtilityFunction`. We fit separate surrogate models at each
    wave as in the original paper, and hence deem a given point to be feasible if it is
    deemed to be feasible by the surrogate models for all waves so far.

    Atributes:
        search_space: The search space over which to optimise the acquisition function.
        posterior_handlers: A dictionary mapping from wave tag to a dictionary mapping
        from principle component tag to the posterior handler for that principle
        component.
        transformed_metrics_datasets: A dictionary mapping from wave tag to a dictionary
        mapping from principle component tag to the dataset for that principle
        component.
        untransformed_metrics_datasets: A dictionary mapping from wave tag to the
        dataset of the untransformed metrics for that wave.
        key: A JAX PRNG key.
        batch_size: The batch size to use for the acquisition function.
        threshold_plausibility: Object for determining which parameter
        configurations are plausible by calling its `get_plausible_points` method.
    """

    search_space: ContinuousSearchSpace
    # We maintain a surrogate model for each wave, and each principle component. As
    # such, `posterior_handlers` and `datasets` take the form of nested dictionaries.
    # e.g. `posterior_handlers["WAVE0"]["PC0"]` is the posterior handler for the
    # surrogate model for wave 0, principle component 0.
    posterior_handlers: Dict[str, Dict[str, PosteriorHandler]]
    transformed_metrics_datasets: Dict[str, Dict[str, Dataset]]
    untransformed_metrics_datasets: Dict[
        str, Dataset
    ]  # Just a dict from wave_num -> dataset i.e. we don't have a separate dataset for each metric.
    key: KeyArray
    batch_size: int
    threshold_plausibility: ThresholdPlausibility

    def __post_init__(self):
        self.transformed_metrics_datasets = copy.copy(
            self.transformed_metrics_datasets
        )  # Ensure initial datasets passed in to DecisionMaker are not mutated from within

        self.untransformed_metrics_datasets = copy.copy(
            self.untransformed_metrics_datasets
        )  # Ensure initial datasets passed in to DecisionMaker are not mutated from within

        # Initialise points for calculating the plausible volume
        self.plausible_volume_query_points = self.search_space.sample(
            10000, key=self.key
        )

        # Initialize posteriors
        self.posteriors: Dict[str, Dict[str, AbstractPosterior]] = {}
        wave_0_tag = WAVE_PREFIX + "0"
        self.posteriors[wave_0_tag] = {}
        wave_0_posterior_handlers = self.posterior_handlers[wave_0_tag]
        wave_0_datasets = self.transformed_metrics_datasets[wave_0_tag]
        for (
            principle_component_tag,
            principle_component_posterior_handler,
        ) in wave_0_posterior_handlers.items():
            self.key, _ = jr.split(self.key)
            self.posteriors[wave_0_tag][
                principle_component_tag
            ] = principle_component_posterior_handler.get_posterior(
                wave_0_datasets[principle_component_tag], optimize=True, key=self.key
            )

    def ask(self, key: KeyArray) -> Float[Array, "N D"]:
        sufficient_plausible_points = False
        num_samples_multiplier = 1
        while not sufficient_plausible_points:
            points = search_space.sample(
                self.num_initial_samples * num_samples_multiplier, key=key
            )  # [N, D]
            plausible_points = self.threshold_plausibility.get_plausible_points(
                points,
                self.posteriors,
                self.transformed_metrics_datasets,
            )
            num_plausible_points = plausible_points.shape[0]
            if num_plausible_points > self.batch_size:
                sufficient_plausible_points = True
            else:
                num_samples_multiplier *= 2
        points_to_return = plausible_points[: self.batch_size]
        return points_to_return

    def tell(
        self,
        transformed_metrics_datasets: Dict[str, Dataset],
        untransformed_metrics_dataset: Dataset,
        wave: int,
        key: KeyArray,
    ) -> None:
        wave_tag = WAVE_PREFIX + str(wave)
        self.transformed_metrics_datasets[wave_tag] = {}
        for (
            principle_component_tag,
            principle_component_dataset,
        ) in transformed_metrics_datasets.items():
            self.transformed_metrics_datasets[wave_tag][
                principle_component_tag
            ] = principle_component_dataset

        self.untransformed_metrics_datasets[wave_tag] = untransformed_metrics_dataset

        self.posteriors[wave_tag] = {}
        for (
            principle_component_tag,
            principle_component_dataset,
        ) in transformed_metrics_datasets.items():
            key, _ = jr.split(key)
            self.posteriors[wave_tag][
                principle_component_tag
            ] = self.posterior_handlers[wave_tag][
                principle_component_tag
            ].get_posterior(
                principle_component_dataset, optimize=True, key=key
            )

    def run(
        self,
        n_waves: int,
        black_box_function_evaluator: Callable[
            [Float[Array, "N D"]], Tuple[Dataset, Dict[str, Dataset]]
        ],
    ) -> Dict[str, Dataset]:
        for wave in range(1, n_waves + 1):
            with open(SAVE_DIR + f"start_wave_{wave}_posteriors.pkl", "wb") as f:
                pickle.dump(self.posteriors, f)
            with open(
                SAVE_DIR + f"start_wave_{wave}_transformed_metrics_datasets.pkl", "wb"
            ) as f:
                pickle.dump(self.transformed_metrics_datasets, f)
            with open(
                SAVE_DIR + f"start_wave_{wave}_untransformed_metrics_datasets.pkl", "wb"
            ) as f:
                pickle.dump(self.untransformed_metrics_datasets, f)
            with open(SAVE_DIR + f"start_wave_{wave}_key.pkl", "wb") as f:
                pickle.dump(self.key, f)
            plausible_points = self.threshold_plausibility.get_plausible_points(
                self.plausible_volume_query_points,
                self.posteriors,
                self.transformed_metrics_datasets,
            )
            feasible_volume = (
                plausible_points.shape[0] / self.plausible_volume_query_points.shape[0]
            )
            with open(SAVE_DIR + f"start_wave_{wave}_plausible_volume.pkl", "wb") as f:
                pickle.dump(feasible_volume, f)
            print(f"Wave {wave} Plausible Volume: {feasible_volume}")
            # Set number of random samples based on plausible volume
            self.num_initial_samples = int(self.batch_size / feasible_volume)
            query_points = self.ask(self.key)

            self.key, _ = jr.split(self.key)
            (
                untransformed_metrics_dataset,
                transformed_metrics_datasets,
            ) = black_box_function_evaluator(query_points)
            self.tell(
                transformed_metrics_datasets=transformed_metrics_datasets,
                untransformed_metrics_dataset=untransformed_metrics_dataset,
                wave=wave,
                key=self.key,
            )

            with open(SAVE_DIR + f"end_wave_{wave}_posteriors.pkl", "wb") as f:
                pickle.dump(self.posteriors, f)
            with open(
                SAVE_DIR + f"end_wave_{wave}_transformed_metrics_datasets.pkl", "wb"
            ) as f:
                pickle.dump(self.transformed_metrics_datasets, f)
            with open(
                SAVE_DIR + f"end_wave_{wave}_untransformed_metrics_datasets.pkl", "wb"
            ) as f:
                pickle.dump(self.untransformed_metrics_datasets, f)
            with open(SAVE_DIR + f"end_wave_{wave}_key.pkl", "wb") as f:
                pickle.dump(self.key, f)

        return self.untransformed_metrics_datasets, self.transformed_metrics_datasets


if __name__ == "__main__":
    onp.random.seed(RANDOM_SEED)

    # Use x_init as in original paper
    l96_x_init = 10 * onp.ones(36)
    l96_x_init[18] = 10 + 0.01
    param_lower_bounds = onp.array([-2.0, -20.0, 0.0, -20.0])
    param_upper_bounds = onp.array([2.0, 20.0, 20.0, 20.0])
    l96_data_processor = Lorenz96DataProcessor(
        X_init=l96_x_init,
        lower_bounds=param_lower_bounds,
        upper_bounds=param_upper_bounds,
    )
    # Params ordered as [h, F, c, b] as in original paper
    l96_true_params = jnp.array(
        [[0.75, 0.75, 0.5, 0.75]]
    )  # Transformed true params into range [0, 1]

    # Search Space is ordered as [h, F, c, b] parameters in original paper
    search_space_lower_bounds = jnp.zeros(4, dtype=jnp.float64)
    search_space_upper_bounds = jnp.ones(4, dtype=jnp.float64)
    search_space = ContinuousSearchSpace(
        lower_bounds=search_space_lower_bounds, upper_bounds=search_space_upper_bounds
    )
    initial_x = onp.array(
        search_space.sample(INITIAL_NUM_SAMPLES, key=jr.PRNGKey(RANDOM_SEED))
    )
    onp_untransformed_initial_y = onp.array(
        Parallel(n_jobs=-2)(
            delayed(l96_data_processor.evaluate_lorenz_96)(x) for x in initial_x
        )
    )
    l96_data_processor.initialise_standardisation_and_pca_transforms(
        onp_untransformed_initial_y
    )
    onp_transformed_initial_y = l96_data_processor.transform_metrics(
        onp_untransformed_initial_y
    )
    jnp_untransformed_initial_y = jnp.array(onp_untransformed_initial_y)
    jnp_transformed_initial_y = jnp.array(onp_transformed_initial_y)
    untransformed_metrics_datasets = {
        WAVE_PREFIX
        + "0": Dataset(X=jnp.array(initial_x), y=jnp_untransformed_initial_y)
    }

    transformed_metrics_datasets = {}
    wave_zero_transformed_metrics_datasets = {}
    for pc in range(l96_data_processor.num_pca_components):
        wave_zero_transformed_metrics_datasets[f"PC{pc}"] = Dataset(
            X=jnp.array(initial_x), y=jnp_transformed_initial_y[:, pc][..., None]
        )
    transformed_metrics_datasets[
        WAVE_PREFIX + "0"
    ] = wave_zero_transformed_metrics_datasets

    (
        untransformed_true_metrics_dataset,
        true_metrics_transformed_datasets,
    ) = l96_data_processor.get_pre_and_post_transformed_metrics(l96_true_params)
    transformed_true_metrics = {
        tag: jnp.squeeze(dataset.y)
        for tag, dataset in true_metrics_transformed_datasets.items()
    }

    with open(SAVE_DIR + "transformed_true_metrics.pkl", "wb") as f:
        pickle.dump(transformed_true_metrics, f)
    with open(SAVE_DIR + "untransformed_true_metrics_dataset.pkl", "wb") as f:
        pickle.dump(untransformed_true_metrics_dataset, f)

    # Initialise Models
    posterior_handlers = {}
    for wave in range(NUM_WAVES + 1):
        wave_posterior_handlers = {}
        for pc in range(l96_data_processor.num_pca_components):
            tag = PC_PREFIX + str(pc)
            mean = gpx.Constant(transformed_true_metrics[tag])
            mean = mean.replace_trainable(constant=TRAINABLE_PRIOR_MEAN)
            kernel = gpx.Matern52(active_dims=[0, 1, 2, 3], lengthscale=jnp.ones(4))
            prior = gpx.Prior(mean_function=mean, kernel=kernel)
            likelihood_builder = lambda x: gpx.Gaussian(
                num_datapoints=x, obs_noise=jnp.array(1e-6)
            )
            posterior_handler = PosteriorHandler(
                prior,
                likelihood_builder=likelihood_builder,
                optimization_objective=gpx.ConjugateMLL(negative=True),
                optimizer=ox.adam(learning_rate=0.01),
                num_optimization_iters=1000,
            )
            wave_posterior_handlers[PC_PREFIX + str(pc)] = posterior_handler
        posterior_handlers[WAVE_PREFIX + str(wave)] = wave_posterior_handlers

    threshold_plausibility = ThresholdPlausibility(
        transformed_true_metrics=transformed_true_metrics,
        threshold=THRESHOLD_PLAUSIBILITY,
    )

    dm = ClimateModelDecisionMaker(
        search_space=search_space,
        posterior_handlers=posterior_handlers,
        transformed_metrics_datasets=transformed_metrics_datasets,
        untransformed_metrics_datasets=untransformed_metrics_datasets,
        batch_size=BATCH_SIZE,
        key=jr.PRNGKey(RANDOM_SEED),
        threshold_plausibility=threshold_plausibility,
    )
    (
        final_untransformed_metrics_datasets,
        final_transformed_metrics_datasets,
    ) = dm.run(
        n_waves=NUM_WAVES,
        black_box_function_evaluator=l96_data_processor.get_pre_and_post_transformed_metrics,
    )
    with open(SAVE_DIR + "lorenz96_data_processor.pkl", "wb") as f:
        pickle.dump(l96_data_processor, f)

    with open(SAVE_DIR + "final_transformed_metrics_datasets.pkl", "wb") as f:
        pickle.dump(final_transformed_metrics_datasets, f)

    with open(SAVE_DIR + "final_untransformed_metrics_datasets.pkl", "wb") as f:
        pickle.dump(final_untransformed_metrics_datasets, f)
