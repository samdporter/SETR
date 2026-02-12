import logging

import numpy as np
import pandas as pd
from cil.framework import BlockDataContainer
from cil.optimisation.utilities import callbacks
from sirf.STIR import ImageData

from recon_core.utils.metrics import compute_all_metrics, compute_block_metrics


class Callback(callbacks.Callback):
    """
    CIL Callback but with `self.skip_iteration` checking `min(self.interval,
    algo.update_objective_interval)`.
    TODO: backport this class to CIL.
    """

    def __init__(self, interval, **kwargs):
        super().__init__(**kwargs)
        self.interval = interval

    def skip_iteration(self, algo) -> bool:
        return (
            algo.iteration % min(self.interval, algo.update_objective_interval) != 0
            and algo.iteration != algo.max_iteration
        )


class SaveImageCallback(Callback):
    """
    CIL Callback that saves an image to disk.
    """

    def __init__(self, filename, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename

    def __call__(self, algo):
        if self.skip_iteration(algo):
            return
        if isinstance(algo.solution, ImageData):
            algo.solution.write(f"{self.filename}_{algo.iteration}.hv")
        elif isinstance(algo.solution, BlockDataContainer):
            for i, el in enumerate(algo.solution.containers):
                el.write(f"{self.filename}_{i}_{algo.iteration}.hv")


class SaveKernelisedImageCallback(Callback):
    """
    Save the alpha image to disk.
    """

    def __init__(self, filename, interval, kernel_op, **kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename
        self.kernel_op = kernel_op

    def __call__(self, algo):
        if algo.iteration % self.interval != 0:
            return
        image = self.kernel_op.recon.compute_kernelised_image(algo.solution, algo.solution)
        image.write(f"{self.filename}_{algo.iteration}.hv")


class SaveGradientUpdateCallback(Callback):
    """
    CIL Callback that saves the gradient update to disk.
    """

    def __init__(self, filename, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename

    def __call__(self, algo):
        if self.skip_iteration(algo):
            return
        if isinstance(algo.gradient_update, ImageData):
            algo.gradient_update.write(f"{self.filename}_{algo.iteration}.hv")
        elif isinstance(algo.gradient_update, BlockDataContainer):
            for i, el in enumerate(algo.gradient_update.containers):
                el.write(f"{self.filename}_{i}_{algo.iteration}.hv")


class PrintObjectiveCallback(Callback):
    """
    CIL Callback that prints the objective function value to the console.
    """

    def __call__(self, algo):
        if algo.iteration % algo.update_objective_interval == 0:
            logging.info(f"iter: {algo.iteration} objective: {algo.objective[-1]}")


class SaveObjectiveCallback(Callback):
    """
    CIL Callback that saves the objective function value to disk.
    """

    def __init__(self, filename, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename

    def __call__(self, algo):
        if self.skip_iteration(algo):
            return
        pd.DataFrame(algo.objective).to_csv(f"{self.filename}.csv")

class SaveStepSizeCallback(Callback):
    """
    Callback to save the step size at each iteration to a CSV file.

    This is useful for analyzing the behavior of adaptive step size rules like
    Armijo line search.

    Args:
        filename (str): Path to the output CSV file.
        interval (int): Save step size every `interval` iterations.
    """

    def __init__(self, filename, interval=1):
        super().__init__(interval)
        self.filename = filename
        self.step_sizes_df = pd.DataFrame(columns=["iteration", "step_size"])

    def __call__(self, algorithm):
        """
        Save the current step size if the interval is met.
        """
        if self.skip_iteration(algorithm):
            return

        iteration = algorithm.iteration
        
        if hasattr(algorithm.step_size_rule, 'get_step_size'):
            step_size = algorithm.step_size_rule.get_step_size()
        else:
            step_size = algorithm.step_size

        # Append new row to DataFrame
        self.step_sizes_df.loc[iteration] = [iteration, step_size]

        # Save the entire DataFrame to CSV
        try:
            self.step_sizes_df.to_csv(f"{self.filename}.csv", index=False)
        except IOError as e:
            logging.error(f"Could not write to step size file: {e}")


class SavePreconditionerCallback(Callback):
    """
    CIL Callback that saves the preconditioner to disk.
    """

    def __init__(self, filename, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename
        self._warned_block_precond = False

    def __call__(self, algo):
        if self.skip_iteration(algo):
            return
        if algo.preconditioner is None:
            return

        preconditioner = algo.preconditioner.compute_preconditioner(algo)

        if isinstance(preconditioner, np.ndarray):
            if not self._warned_block_precond:
                logging.warning(
                    "SavePreconditionerCallback: block/array preconditioners are not serialised to .hv; skipping save."
                )
                self._warned_block_precond = True
            return

        if isinstance(preconditioner, ImageData):
            preconditioner.write(f"{self.filename}_{algo.iteration}.hv")
        elif isinstance(preconditioner, BlockDataContainer):
            for i, el in enumerate(preconditioner.containers):
                el.write(f"{self.filename}_{i}_{algo.iteration}.hv")
        else:
            logging.warning(
                "SavePreconditionerCallback: unsupported preconditioner type %s; skipping save.",
                type(preconditioner).__name__,
            )


class SubsetValueCallback(Callback):
    """
    CIL Callback that saves the stochastic gradient value to disk.
    """

    def __init__(self, filename, interval, **kwargs):
        super().__init__(interval, **kwargs)
        self.filename = filename
        # create panda dataframe and save all subset fucntion values in it
        self.subset_values = pd.DataFrame()

    def __call__(self, algo):
        if self.skip_iteration(algo):
            return
        try:
            func_list = algo.f.functions
        except AttributeError:
            func_list = algo.f.function.functions
        for i, function in enumerate(func_list):
            # needs to add to new line for iteration algo.iteration
            self.subset_values.at[algo.iteration, f"Subset {i}"] = function(algo.solution)
        # add a sum at first column
        self.subset_values.at[algo.iteration, "Sum"] = sum(
            function(algo.solution) for function in func_list
        )
        self.subset_values.to_csv(f"{self.filename}.csv")


class ComputeMetricsCallback(Callback):
    """
    CIL Callback that computes image quality metrics against a reference image.

    Computes MSE, RMSE, NRMSE, MAE, NMAE at specified intervals and saves to CSV.
    Supports both single images and BlockDataContainer (multi-modal).
    Optionally applies a mask to restrict metrics to region of interest.

    Args:
        reference: Reference image (ImageData or BlockDataContainer)
        filename: Output CSV filename (without extension)
        interval: Compute metrics every N iterations
        mask: Optional mask (ImageData, BlockDataContainer, or None)
        normalization: Normalization method for NRMSE/NMAE ('range', 'max', 'mean', 'euclidean')
        verbose: If True, log metrics to console

    Example:
        >>> reference = ImageData("ground_truth.hv")
        >>> mask = create_mask_from_threshold(reference, threshold=0.1)
        >>> callback = ComputeMetricsCallback(
        ...     reference=reference,
        ...     filename="output/metrics",
        ...     interval=10,
        ...     mask=mask,
        ...     verbose=True
        ... )
        >>> algo.run(100, callbacks=[callback])
    """

    def __init__(
        self,
        reference,
        filename,
        interval,
        mask=None,
        normalization="range",
        verbose=False,
        **kwargs,
    ):
        super().__init__(interval, **kwargs)
        self.reference = reference
        self.filename = filename
        self.mask = mask
        self.normalization = normalization
        self.verbose = verbose
        self.metrics_df = pd.DataFrame()

        # Determine if we're working with BlockDataContainer
        self.is_block = isinstance(reference, BlockDataContainer)

    def __call__(self, algo):
        if self.skip_iteration(algo):
            return

        iteration = algo.iteration

        if self.is_block:
            # Multi-modal metrics
            metrics = compute_block_metrics(
                algo.solution, self.reference, self.mask, self.normalization
            )

            # Flatten nested dictionary for DataFrame
            # Format: modality_0_mse, modality_0_rmse, modality_1_mse, etc.
            flat_metrics = {}
            for modality, mod_metrics in metrics.items():
                for metric_name, value in mod_metrics.items():
                    flat_metrics[f"{modality}_{metric_name}"] = value

            # Add to DataFrame
            for key, value in flat_metrics.items():
                self.metrics_df.at[iteration, key] = value

            if self.verbose:
                logging.info(f"Iteration {iteration} metrics:")
                for modality, mod_metrics in metrics.items():
                    logging.info(
                        f"  {modality}: "
                        f"RMSE={mod_metrics['rmse']:.6e}, "
                        f"NRMSE={mod_metrics['nrmse']:.6f}"
                    )
        else:
            # Single image metrics
            metrics = compute_all_metrics(
                algo.solution, self.reference, self.mask, self.normalization
            )

            # Add to DataFrame
            for key, value in metrics.items():
                self.metrics_df.at[iteration, key] = value

            if self.verbose:
                logging.info(
                    f"Iteration {iteration} metrics: "
                    f"RMSE={metrics['rmse']:.6e}, "
                    f"NRMSE={metrics['nrmse']:.6f}"
                )

        # Save to CSV
        self.metrics_df.to_csv(f"{self.filename}.csv", index_label="iteration")


class PrintMetricsCallback(Callback):
    """
    CIL Callback that computes and prints image quality metrics to console only.

    Lighter version of ComputeMetricsCallback that doesn't save to disk.
    Useful for quick monitoring during reconstruction.

    Args:
        reference: Reference image (ImageData or BlockDataContainer)
        interval: Compute metrics every N iterations
        mask: Optional mask (ImageData, BlockDataContainer, or None)
        normalization: Normalization method for NRMSE/NMAE
        metrics: List of metrics to print (default: ['rmse', 'nrmse'])
    """

    def __init__(
        self, reference, interval, mask=None, normalization="range", metrics=None, **kwargs
    ):
        super().__init__(interval, **kwargs)
        self.reference = reference
        self.mask = mask
        self.normalization = normalization
        self.metrics_to_print = metrics or ["rmse", "nrmse"]
        self.is_block = isinstance(reference, BlockDataContainer)

    def __call__(self, algo):
        if self.skip_iteration(algo):
            return

        iteration = algo.iteration

        if self.is_block:
            metrics = compute_block_metrics(
                algo.solution, self.reference, self.mask, self.normalization
            )

            logging.info(f"Iteration {iteration} metrics:")
            for modality, mod_metrics in metrics.items():
                metric_str = ", ".join(
                    f"{m.upper()}={mod_metrics[m]:.6e}"
                    for m in self.metrics_to_print
                    if m in mod_metrics
                )
                logging.info(f"  {modality}: {metric_str}")
        else:
            metrics = compute_all_metrics(
                algo.solution, self.reference, self.mask, self.normalization
            )

            metric_str = ", ".join(
                f"{m.upper()}={metrics[m]:.6e}" for m in self.metrics_to_print if m in metrics
            )
            logging.info(f"Iteration {iteration} metrics: {metric_str}")
#
