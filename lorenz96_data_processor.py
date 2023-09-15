from jax import config

config.update("jax_enable_x64", True)

from dataclasses import dataclass

from beartype.typing import (
    Dict,
    Tuple,
)
import jax.numpy as jnp
from jaxtyping import Float
from joblib import (
    Parallel,
    delayed,
)
import numpy as onp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from gpjax.dataset import Dataset
from gpjax.typing import (
    Array,
    Float,
)
from lorenz96 import L96TwoLevelOriginal


@dataclass
class Lorenz96DataProcessor:
    """
    A class for processing data from the Lorenz 96 model. Most code based on that found
    in the codebase accompanying the original paper, with key code found here:
    https://github.com/HRMES-MOPGA/L96HistoryMatching/blob/main/GenerateData_Python/Exp1-HM-TuningL96-Python.ipynb
    """

    X_init: onp.ndarray  # Shape is [36, ]
    lower_bounds: onp.ndarray  # Shape is [4, ]
    upper_bounds: onp.ndarray  # Shape is [4, ]

    def evaluate_lorenz_96(self, param_configuration: onp.ndarray) -> onp.ndarray:
        """
        Run a simulation of the Lorenz 96 model from `self.X_init` with the given
        parameters and return the mean statistics of the simulation.

        Args:
            param_configuration: The parameters to use for the simulation. Shape is [4,
            ].
        Returns:
            The mean statistics of the simulation. Shape is [180, ].
        """
        # Transform the parameters from [0, 1] range used for fitting GPs to the actual parameter ranges
        transformed_param_configuration = self.lower_bounds + param_configuration * (
            self.upper_bounds - self.lower_bounds
        )
        l96param_spinup = L96TwoLevelOriginal(
            K=36,
            save_dt=0.001,
            X_init=self.X_init,
            h=transformed_param_configuration[0],
            F=transformed_param_configuration[1],
            c=transformed_param_configuration[2],
            b=transformed_param_configuration[3],
            integration_type="coupled",
        )
        l96param_spinup.iterate(10)
        l96param = L96TwoLevelOriginal(
            K=36,
            save_dt=0.001,
            X_init=l96param_spinup.history.X[-1, :].values,
            h=transformed_param_configuration[0],
            F=transformed_param_configuration[1],
            c=transformed_param_configuration[2],
            b=transformed_param_configuration[3],
            integration_type="coupled",
        )
        l96param.iterate(100)
        onp_stats = l96param.mean_stats(ax=0)
        return onp_stats

    def initialise_standardisation_and_pca_transforms(self, onp_y: onp.ndarray):
        """
        Initialise the standardisation and PCA transforms for the given data. The
        statistics produced by the Lorenz96 simulator are of shape [180, ]. Lguensat et
        al. (https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2022MS003367)
        propose using PCA to reduce the dimensionality of the statistics. Therefore, we
        use PCA. Before applying PCA, we standardise the data (as is required for PCA),
        and we then standardise the PCA-transformed data for fitting the GPs. We use the
        initial samples to fit the standardisation and PCA transforms. The number of PCA
        components is chosen to be the minimum number of components that explain 99% of
        the variance in the data.

        Args:
            onp_y: The initial data to use for fitting the standardisation and PCA transforms. Shape is
            [N, 180].
        """
        self.pre_pca_scaler = StandardScaler().fit(onp_y)
        onp_y_scaled = self.pre_pca_scaler.transform(onp_y)
        self.pca = PCA(n_components=0.99, svd_solver="full").fit(onp_y_scaled)
        pca_transformed_y = self.pca.transform(onp_y_scaled)
        self.num_pca_components = pca_transformed_y.shape[1]
        self.post_pca_scaler = StandardScaler().fit(pca_transformed_y)

    def transform_metrics(self, untransformed_metrics: onp.ndarray) -> onp.ndarray:
        """
        Take the given untransformed metrics and return the PCA-transformed metrics.

        Args:
            untransformed_metrics: The untransformed metrics to transform. Shape is [N,
            180].
        Returns:
            The PCA-transformed metrics. Shape is [N, PCA_COMPONENTS].
        """
        scaled_metrics = self.pre_pca_scaler.transform(untransformed_metrics)
        pca_transformed_metrics = self.pca.transform(scaled_metrics)
        pca_transformed_metrics = self.post_pca_scaler.transform(
            pca_transformed_metrics
        )
        return pca_transformed_metrics

    def get_pre_and_post_transformed_metrics(
        self, param_configurations: Float[Array, "N 4"]
    ) -> Tuple[Dataset, Dict[str, Dataset]]:
        """
        Take a set of parameter configurations and run the Lorenz96 simulator for each
        of them. Return the untransformed and transformed metrics for each simulation.
        Note that the arguments and return values are in JAX format, but are converted
        to ordinary numpy arrays for compatibility with the Lorenz96 simulator.

        Args:
            param_configurations: The parameter configurations to use for the
            simulations. Shape is [N, 4].
        Returns:
            The untransformed metrics dataset and the transformed metrics datasets. The
            untransformed metrics dataset has shape [N, 180]. The transformed metrics
            datasets have shape [N, PCA_COMPONENTS].
        """
        onp_param_configurations = onp.array(param_configurations)
        num_configurations = onp_param_configurations.shape[0]
        if num_configurations > 1:
            # Use all but one CPUs to run the simulations in parallel
            onp_untransformed_metrics = Parallel(n_jobs=-2)(
                delayed(self.evaluate_lorenz_96)(onp_param_config)
                for onp_param_config in onp_param_configurations
            )
            onp_untransformed_metrics = onp.array(onp_untransformed_metrics)
        else:
            onp_untransformed_metrics = self.evaluate_lorenz_96(
                onp_param_configurations[0]
            )
            onp_untransformed_metrics = onp_untransformed_metrics[None, ...]

        if not onp.isfinite(onp_untransformed_metrics).all():
            raise ValueError("Non-finite values encountered in simulation metrics.")
        jnp_untransformed_metrics = jnp.array(onp_untransformed_metrics)
        onp_transformed_metrics = self.transform_metrics(onp_untransformed_metrics)
        jnp_transformed_metrics = jnp.array(onp_transformed_metrics)
        untransformed_metrics_dataset = Dataset(
            X=param_configurations, y=jnp_untransformed_metrics
        )
        transformed_metrics_datasets = {}
        for i in range(self.num_pca_components):
            transformed_metrics_datasets[f"PC{i}"] = Dataset(
                X=param_configurations,
                y=jnp_transformed_metrics[:, i][..., None],
            )
        return untransformed_metrics_dataset, transformed_metrics_datasets
