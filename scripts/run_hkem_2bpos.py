#!/usr/bin/env python3
"""SETR HKEM reconstruction for multiple bed positions - Simplified version using shared modules."""

import cProfile
import logging
import os
import pstats
from types import SimpleNamespace

from cil.optimisation.algorithms import ISTA
from cil.optimisation.functions import (
    OperatorCompositionFunction,
    SGFunction,
    SumFunction,
)
from cil.optimisation.operators import CompositionOperator
from cil.optimisation.utilities import Sampler
from sirf.contrib.partitioner import partitioner

from setr.cil_extensions.algorithms import ista_update_step
from setr.cil_extensions.callbacks import (
    PrintObjectiveCallback,
    SaveGradientUpdateCallback,
    SaveImageCallback,
    SaveObjectiveCallback,
    SavePreconditionerCallback,
)
from setr.cil_extensions.framework.framework import EnhancedBlockDataContainer
from setr.cil_extensions.functions import BlockIndicatorBox
from setr.cil_extensions.operators import AdjointOperator, CouchShiftOperator, TruncationOperator
from setr.cil_extensions.preconditioners import (
    SubsetKernelisedEMPreconditioner,
)
from setr.scripts.common import (
    apply_combine_sensitivities,
    configure_logging,
    get_shift_operators,
    init_run_env,
)
from setr.scripts.hkem_common import get_kernel_hyperparams, get_kernel_operator
from setr.utils import get_pet_data_multiple_bed_pos
from setr.utils.io import apply_overrides, load_config, parse_cli, save_args
from setr.utils.sirf import get_pet_am

DEBUG = False

if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
    ISTA.update = ista_update_step  # Patch ISTA with our custom update step
    logging.info("Debug mode is ON. More intermediate results will be saved.")
else:
    logging.basicConfig(level=logging.INFO)
    logging.info("Debug mode is OFF.")


def prepare_data(args):
    """Prepare the multi-bed PET data and guidance image."""
    pet_data = get_pet_data_multiple_bed_pos(
        args.pet_data_path, tof=args.use_tof, suffixes=["_f1b1", "_f2b1"]
    )

    # Create kernel operators for each bed position
    if args.guidance == "attenuation":
        guidance = pet_data["attenuation"]
    elif args.guidance == "emission":
        guidance = pet_data["spect"]
        assert guidance is not None, "Emission guidance selected but no SPECT data provided"
    else:
        raise ValueError(f"Unknown guidance type: {args.guidance}")
    assert type(guidance) is type(pet_data["template_image"]), (
        f"Guidance and initial estimates must be same type."
        f"Got {type(guidance)} and {type(pet_data['initial_image'])}"
    )

    if DEBUG:
        print(
            f"shape of guidance: {guidance.shape}, "
            f"initial_estimates: {pet_data['initial_image'].shape}"
        )

    return pet_data, guidance


def run_hkem_ista(args, pet_data, guidance, hyperparams):
    """Run ISTA-based HKEM reconstruction with kernel preconditioner."""

    # Get acquisition model function
    def get_am():
        return get_pet_am(gpu=not args.no_gpu, gauss_fwhm=args.pet_gauss_fwhm)

    pet_dfs = [
        partitioner.data_partition(
            pet_data["bed_positions"][suffix]["acquisition_data"],
            pet_data["bed_positions"][suffix]["additive"],
            pet_data["bed_positions"][suffix]["normalisation"],
            num_batches=args.num_subsets,
            mode="staggered",
            create_acq_model=get_am,
        )[2]
        for suffix in pet_data["bed_positions"]
    ]

    # Set up subset objectives on their own bed template
    for i, suffix in enumerate(pet_data["bed_positions"]):
        tmpl = pet_data["bed_positions"][suffix]["template_image"]
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j].set_up(tmpl)

    # Get sensitivities for each bed position
    pet_sens = [[f.get_subset_sensitivity(0).maximum(0) for f in df] for df in pet_dfs]

    # Set up shift operators
    uncombine_op, unshift_ops, choose_ops = get_shift_operators(pet_data)

    bed_sens = [sum(sens_list) for sens_list in pet_sens]
    apply_combine_sensitivities(pet_data, bed_sens)

    shift = CouchShiftOperator.get_couch_shift_from_sinogram(
        pet_data["bed_positions"]["_f2b1"]["acquisition_data"]
    )
    print(f"Using shift of {shift}mm for bed position 2")
    zero_shift_op = CouchShiftOperator(pet_data["template_image"], 0)
    unzero_shift_op = AdjointOperator(zero_shift_op)

    # Combine corresponding subsets across bed positions
    sensitivities = [
        # scale by num_subsets
        args.num_subsets
        * unzero_shift_op.adjoint(
            uncombine_op.adjoint(
                EnhancedBlockDataContainer(
                    *[
                        unshift_op.adjoint(sens[subset_idx])
                        for unshift_op, sens in zip(unshift_ops, pet_sens)
                    ]
                )
            )
        )
        for subset_idx in range(args.num_subsets)
    ]

    kernel = get_kernel_operator(
        args,
        zero_shift_op.direct(guidance),
        zero_shift_op.direct(pet_data["template_image"]),
        pet_data["bed_positions"]["_f1b1"]["acquisition_data"],
        hyperparams,
    )

    # Wrap PET objectives with uncombine/choose/unshift operators
    for i, suffix in enumerate(pet_data["bed_positions"]):
        for j in range(len(pet_dfs[i])):
            pet_dfs[i][j] = OperatorCompositionFunction(
                pet_dfs[i][j],
                CompositionOperator(
                    unshift_ops[i],
                    choose_ops[i],
                    uncombine_op,
                    unzero_shift_op,
                ),
            )

    # Combine objectives across bed positions
    f_list = [
        OperatorCompositionFunction(SumFunction(*[funs[j] for funs in pet_dfs]), kernel)
        for j in range(args.num_subsets)
    ]

    sampler = Sampler.sequential(args.num_subsets)
    f = -SGFunction(f_list, sampler)
    g = BlockIndicatorBox(lower=0)

    if DEBUG:
        for i, s in enumerate(sensitivities):
            s.write(os.path.join(args.output_path, f"sens_before_kernel_{i}.hv"))
            s2 = kernel.direct(s)
            s2.write(os.path.join(args.output_path, f"sens_after_kernel_{i}.hv"))

    # Create preconditioner
    precond = SubsetKernelisedEMPreconditioner(
        args.num_subsets,
        sensitivities,
        kernel,
        freeze_iter=args.freeze_iter,
        epsilon=pet_data["template_image"].max() * 1e-12,
    )

    # Initialize alpha
    truncate = TruncationOperator(pet_data["template_image"])
    init_alpha = zero_shift_op.direct(pet_data["template_image"].get_uniform_copy(1))
    init_alpha = truncate.direct(init_alpha)  # Apply truncation

    # Set up algorithm
    algo = ISTA(
        init_alpha,
        f,
        g,
        step_size=args.step_size,
        preconditioner=precond,
        update_objective_interval=args.num_subsets,
    )

    # Set up callbacks
    class SaveKernelisedImageCallback:
        """Save the kernelised image (x = K(alpha)) to disk."""

        def __init__(self, filename, interval, kernel_op):
            self.filename = filename
            self.interval = interval
            self.kernel_op = kernel_op

        def __call__(self, algo):
            if algo.iteration % self.interval != 0:
                return
            # Save the kernelised image
            image = self.kernel_op.direct(algo.solution)
            image.write(f"{self.filename}_{algo.iteration}.hv")

    interval = 1 if DEBUG else args.num_subsets

    callbacks = [
        SaveImageCallback(os.path.join(args.output_path, "alpha"), interval=interval),
        SaveKernelisedImageCallback(
            os.path.join(args.output_path, "x"), interval=interval, kernel_op=kernel
        ),
        PrintObjectiveCallback(interval=args.num_subsets),
        SaveObjectiveCallback(os.path.join(args.output_path, "objective"), interval=interval),
    ]

    if DEBUG:
        callbacks.extend(
            [
                SavePreconditionerCallback(
                    os.path.join(args.output_path, "preconditioner"), interval=interval
                ),
                SaveGradientUpdateCallback(
                    os.path.join(args.output_path, "gradient"), interval=interval
                ),
            ]
        )

    logging.info("Running HKEM-ISTA reconstruction...")
    num_subiterations = args.num_epochs * args.num_subsets
    algo.run(num_subiterations, callbacks=callbacks, verbose=True)

    # Get final results
    output_alpha = algo.solution

    # Apply first kernel to get kernelised image
    output_x = kernel.direct(output_alpha)

    # Save final results
    output_alpha.write(os.path.join(args.output_path, "reconstruction_alpha.hv"))
    output_x.write(os.path.join(args.output_path, "reconstruction_x.hv"))

    return output_alpha, output_x


def main(args):
    """Main function to run HKEM multi-bed reconstruction."""
    configure_logging()

    # Save arguments
    save_args(args, "hkem_2bpos_args.csv")

    logging.info(f"Starting HKEM {args.method.upper()} reconstruction")
    logging.info(f"Modality: {args.modality}")
    logging.info(f"Guidance: {args.guidance}")

    # Prepare data
    pet_data, guidance = prepare_data(args)

    hyperparams = get_kernel_hyperparams(args)

    # Run reconstruction (only ISTA supported for multi-bed)
    output_alpha, output_x = run_hkem_ista(args, pet_data, guidance, hyperparams)

    logging.info("HKEM multi-bed reconstruction completed successfully")
    logging.info(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    # Parse arguments and configuration
    cli = parse_cli()
    config = load_config(cli.config)
    config = apply_overrides(config, cli.override)
    args = SimpleNamespace(**config)

    # Initialize run environment
    msg = init_run_env(args)

    if getattr(args, "profile", True):
        logging.info("Profiling is enabled. This may slow down the execution.")
        profiler = cProfile.Profile()
        profiler.enable()

        main(args)

        profiler.disable()
        profiler.dump_stats(f"{args.output_path}/profile_data.prof")
        # Output results to a file
        output_file = os.path.join(args.output_path, "profiling_results.txt")
        with open(output_file, "w") as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.strip_dirs().sort_stats("cumulative").print_stats()
        logging.info(f"Profiling results saved to {output_file}")
    else:
        logging.info("Profiling is disabled.")
        main(args)
    logging.info("Execution completed.")
