from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from . import load_processed, utils


def plot_depth_histogram(
    mod_file_names: list[str | Path],
    regions_list: list[str | Path | list[str | Path]],
    motifs: list[str],
    sample_names: list[str],
    window_size: int | None = None,
    single_strand: bool = False,
    one_depth_per_region: bool = False,
    cores=None,
    **kwargs,
) -> Axes:
    """
    Plot depth histograms, overlaying the results on top of each other.

    Each input list is expected to be parallel and the same length. Each index represents one analysis condition across the lists.
    Using the same file for multiple conditions requires adding the same file multiple times, in the appropriate indices.

    This is the most flexible method for depth histogram plotting. For most use cases, consider
    using one of the plot_depth_histogram.by_* methods.

    Args:
        mod_file_names: list of paths to modified base data files
        bed_file_names: list of paths to bed files specifying centered equal-length regions
        mod_names: list of modifications to extract; expected to match mods available in the relevant mod_files
        sample_names: list of names to use for labeling traces in the output; legend entries
        window_size: half-size of the desired window to plot; how far the window stretches on either side of the center point
        single_strand: True means we only grab counts from reads from the same strand as
            the region of interest, False means we always grab both strands within the regions
        one_depth_per_region: if True, each region will only report a single depth value, averaging across all non-zero depths. If False
            depths will be reported separately for all nonzero count positions in each region for a more granular view of depth distribution.
        kwargs: other keyword parameters passed through to utils.line_plot

    Returns:
        Axes object containing the plot
    """
    if not utils.check_len_equal(mod_file_names, regions_list, motifs, sample_names):
        raise ValueError("Unequal number of inputs")

    depth_vectors = get_depth_counts(
        mod_file_names=mod_file_names,
        regions_list=regions_list,
        motifs=motifs,
        window_size=window_size,
        single_strand=single_strand,
        one_depth_per_region=one_depth_per_region,
        cores=cores,
    )

    axes = make_depth_histogram_plot(
        depth_vectors=depth_vectors,
        sample_names=sample_names,
        one_depth_per_region=one_depth_per_region,
        y_label="regions count" if one_depth_per_region else "positions count",
        **kwargs,
    )
    return axes


def by_modification(
    mod_file_name: str | Path,
    regions: str | Path,
    motifs: list[str],
    **kwargs,
) -> Axes:
    """
    Plot depth histogram, holding modification file and regions constant, varying modification types

    See plot_depth_histogram for details.
    """
    n_mods = len(motifs)
    return plot_depth_histogram(
        mod_file_names=[mod_file_name] * n_mods,
        regions_list=[regions] * n_mods,
        motifs=motifs,
        sample_names=[f"{motif} depth" for motif in motifs],
        **kwargs,
    )


def by_regions(
    mod_file_name: str | Path,
    regions_list: list[str | Path | list[str | Path]],
    motif: str,
    sample_names: list[str] | None = None,
    **kwargs,
) -> Axes:
    """
    Plot depth histogram, holding modification file and modification types constant, varying regions

    Note: Sample names default to the names of the bed files.

    See plot_depth_histogram for details.
    """
    if sample_names is None:
        sample_names = regions_list
    n_beds = len(regions_list)
    return plot_depth_histogram(
        mod_file_names=[mod_file_name] * n_beds,
        regions_list=regions_list,
        motifs=[motif] * n_beds,
        sample_names=[f"{sample_name} depth" for sample_name in sample_names],
        **kwargs,
    )


def by_dataset(
    mod_file_names: list[str | Path],
    regions: str | Path | list[str | Path],
    motif: str,
    sample_names: list[str] | None = None,
    **kwargs,
) -> Axes:
    """
    Plot depth histogram, holding modification types and regions constant, varying modification files

    Note: Sample names default to the names of the modification files.

    See plot_depth_histogram for details.
    """
    if sample_names is None:
        sample_names = mod_file_names
    n_mod_files = len(mod_file_names)
    return plot_depth_histogram(
        mod_file_names=mod_file_names,
        regions_list=[regions] * n_mod_files,
        motifs=[motif] * n_mod_files,
        sample_names=[f"{sample_name} depth" for sample_name in sample_names],
        **kwargs,
    )


def get_depth_counts(
    mod_file_names: list[str | Path],
    regions_list: list[str | Path | list[str | Path]],
    motifs: list[str],
    window_size: int | None,
    single_strand: bool = False,
    one_depth_per_region: bool = False,
    cores=1,
) -> list[np.ndarray]:
    """
    Get the depth counts, ready for plotting.

    This helper function can be useful during plot prototyping, when repeatedly building plots from the same data.
    Its outputs can be passed as the first argument to make_depth_histogram_plot().

    Args:
        mod_file_names: list of paths to modified base data files
        bed_file_names: list of paths to bed files specifying centered equal-length regions
        mod_names: list of modifications to extract; expected to match mods available in the relevant mod_files
        window_size: half-size of the desired window to plot; how far the window stretches on either side of the center point
        single_strand: True means we only grab counts from reads from the same strand as
            the region of interest, False means we always grab both strands within the regions
        one_depth_per_region: if True, each region will only report a single depth value, averaging across all non-zero depths. If False
            depths will be reported separately for all nonzero count positions in each region for a more granular view of depth distribution.

    Returns:
        List of depth vectors for histogram
    """
    if not utils.check_len_equal(mod_file_names, regions_list, motifs):
        raise ValueError("Unequal number of inputs")
    # TODO: redefinition error; still need to figure out how to do this elegantly in a way mypy likes
    # dimelo/plot_depth_histogram.py:53: error: Item "str" of "str | Path" has no attribute "suffix"  [union-attr]
    mod_file_names = [Path(fn) for fn in mod_file_names]

    depth_vectors = []
    for mod_file, regions, motif in zip(mod_file_names, regions_list, motifs):
        match mod_file.suffix:
            case ".gz":
                pileup_vectors_list = load_processed.regions_to_list(
                    function_handle=load_processed.pileup_vectors_from_bedmethyl,
                    bedmethyl_file=mod_file,
                    regions=regions,
                    motif=motif,
                    window_size=window_size,
                    single_strand=single_strand,
                    cores=cores,
                )
                # places where read depth is zero are assumed to not have the motif present - this may not always be true,
                # but with the available information in a pileup file it's the best we can do
                read_depth_vectors_list = [
                    valid_base_counts[valid_base_counts > 0]
                    for _, valid_base_counts in pileup_vectors_list
                ]
                if one_depth_per_region:
                    # each region's read depth vector gets collapsed to a single mean value
                    read_depths = np.array(
                        [
                            np.mean(read_depth_vector)
                            for read_depth_vector in read_depth_vectors_list
                        ]
                    )
                else:
                    # each region's read depth vector gets added to one extending read depths list without aggregating
                    read_depths = np.concatenate(read_depth_vectors_list)

            case ".fake":
                read_depths = load_processed.vector_from_fake(
                    mod_file=mod_file,
                    bed_file=regions,
                    motif=motif,
                    window_size=window_size,
                )
            case _:
                raise ValueError(f"Unsupported file type for {mod_file}")
        depth_vectors.append(read_depths)
    return depth_vectors


def make_depth_histogram_plot(
    depth_vectors: list[np.ndarray],
    sample_names: list[str],
    y_label: str = "count",
    one_depth_per_region: bool = False,
    **kwargs,
) -> Axes:
    """
    Plot the given depth histogram traces.

    This helper function can be useful during plot prototyping, when repeatedly building plots from the same data.
    The first argument should be the output of get_depth_histograms().

    Args:
        depth_vectors: list of depth histogram counts
        sample_names: list of names to use for labeling traces in the output; legend entries
        one_depth_per_region: if True, each region will only report a single depth value, averaging across all non-zero depths. If False
            depths will be reported separately for all nonzero count positions in each region for a more granular view of depth distribution.
        kwargs: other keyword parameters passed through to utils.line_plot

    Returns:
        Axes object containing the plot
    """
    if not utils.check_len_equal(depth_vectors, sample_names):
        raise ValueError("Unequal number of inputs")
    x_label = (
        "per strand read\ndepth in region"
        if one_depth_per_region
        else "per strand read\ndepth per position"
    )
    axes = utils.hist_plot(
        value_vectors=depth_vectors,
        value_names=sample_names,
        x_label=x_label,
        y_label=y_label,
        integer_values=not one_depth_per_region,
        **kwargs,
    )
    return axes
