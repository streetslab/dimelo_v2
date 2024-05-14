from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

# This provides the mapping of canonical bases to sets of valid mode names
BASEMOD_NAMES_DICT = defaultdict(lambda: set())
BASEMOD_NAMES_DICT.update(
    {
        "A": {"a", "Y"},
        "C": {"m", "Z"},
    }
)

DEFAULT_COLORS = defaultdict(lambda: "grey")
DEFAULT_COLORS.update(
    {
        "A,0": "blue",
        "A,0,a": "blue",
        "CG,0": "orange",
        "CG,0,m": "yellow",
        "CG,0,h": "red",
        "GCH,1": "purple",
    }
)


class ParsedMotif:
    def __init__(self, motif_string):
        parts = motif_string.split(",")
        if len(parts) == 2:
            # If a mod code isn't specified, we use the default set
            self.motif_seq = parts[0]
            self.modified_pos = int(parts[1])
            if self.modified_pos >= len(self.motif_seq):
                raise ValueError(f"Motif {motif_string} has an out-of-range mod index.")
            self.modified_base = self.motif_seq[self.modified_pos]
            self.mod_codes = BASEMOD_NAMES_DICT[self.modified_base]
            self.warning = None
            # Removed this because it is annoying to have the output all the time. Can bring back easily.
        #             self.warning = f"""
        # WARNING: {motif_string} does not specify mod code.
        # Defaulting to {self.mod_codes}.
        # Processing assumes at most one of {self.mod_codes} is present in your data.
        #                             """
        elif len(parts) == 3:
            # If a mod code is specified, that will be the only one we look for
            self.motif_seq = parts[0]
            self.modified_pos = int(parts[1])
            if self.modified_pos >= len(self.motif_seq):
                raise ValueError(f"Motif {motif_string} has an out-of-range mod index.")
            self.modified_base = self.motif_seq[self.modified_pos]
            self.mod_codes = set(parts[2])
            self.warning = None
        else:
            # Motifs need both a sequence and an index, separated by a comma
            raise ValueError(
                f"Motif {motif_string} must have 2 or 3 comma-separated elements: sequence, index, and (optionally) mod code."
            )


def adjust_threshold(
    thresh,
    quiet=True,
):
    if thresh > 0:
        if thresh > 1:
            if not quiet:
                print(
                    f"Modification threshold of {thresh} assumed to be for range 0-255. {thresh}/255={thresh/255} will be sent to modkit."
                )
            thresh_scaled = thresh / 255
        else:
            if not quiet:
                print(
                    f"Modification threshold of {thresh} will be treated as coming from range 0-1."
                )
            thresh_scaled = thresh

        return thresh_scaled
    return thresh


def regions_dict_from_input(
    regions: str | Path | list[str | Path] | None = None,
    window_size: int | None = None,
) -> dict:
    # TODO: Why is this declared out here, and not within add_region_to_dict? To my eye, that method should just return the fully-loaded dict.
    # I don't think this approach works because add_region_to_dict can be called many times; the regions parameter can be a single bed path / string OR many in a list
    regions_dict: defaultdict[str, list] = defaultdict(list)

    if window_size is not None and window_size <= 0:
        raise ValueError(
            "Invalid window_size. To disable windowing, set window_size to None or do not pass a value (the default is None)."
        )

    if isinstance(regions, list):
        for region in regions:
            add_region_to_dict(region, window_size, regions_dict)
    else:
        add_region_to_dict(regions, window_size, regions_dict)
    for chrom in regions_dict:
        regions_dict[chrom].sort(key=lambda x: x[0])

    return regions_dict


def add_region_to_dict(
    region: str | Path,
    window_size: int | None,
    regions_dict: dict,
):
    # TODO: The flow of this is very confusing, creating mypy errors, and possibly creates actual errors.
    # mypy error: dimelo/utils.py:110: error: Item "str" of "str | Path" has no attribute "name"  [union-attr]
    # Basically, this method is confusing because the string can be a pathlike or a region string.
    # Find a different way to check whether the string is pathlike or a region string, coerce paths to Path objects, then clean up everything else.

    # Added None as a window_size option, it was already handled below so was always a valid input

    # We check whether the region is a path to a .bed file by seeing if, when coerced into a Path object, it has the suffix 'bed'
    if Path(region).suffix == ".bed":
        with open(region) as bed_regions:
            for line_index, line in enumerate(bed_regions):
                fields = line.split()
                if len(fields) > 2:
                    # Per the bed spec, the 6th column is strand
                    # https://genome.ucsc.edu/FAQ/FAQformat.html
                    if len(fields) > 5:
                        chrom, start, end, strand = (
                            fields[0],
                            int(fields[1]),
                            int(fields[2]),
                            fields[5],
                        )
                    # If strand isn't in the bed file, we set to . (neither/both)
                    else:
                        chrom, start, end, strand = (
                            fields[0],
                            int(fields[1]),
                            int(fields[2]),
                            ".",
                        )
                    if window_size is None:
                        regions_dict[chrom].append((start, end, strand))
                    else:
                        center_coord = (start + end) // 2
                        regions_dict[chrom].append(
                            (
                                center_coord - window_size,
                                center_coord + window_size,
                                strand,
                            )
                        )
                else:
                    raise ValueError(
                        f"Invalid bed format line {line_index} of {Path(region).name}"
                    )
    # If the region is a path but *not* to a bed file, that isn't valid
    elif isinstance(region, Path):
        raise ValueError(
            f"Path object {region} is not pointing to a .bed file. regions must be provided as paths to .bed files or as strings in the format chrX:XXX-XXX,strand."
        )
    # If the region is a string and doesn't convert to a path to a bed file, then it must be a region string else it cannot be parsed
    elif (
        isinstance(region, str)
        and len(region.split(":")) == 2
        and 2 <= len(region.split(":")[1].split("-")) <= 3
    ):
        # region strings can be either chrX:XXX-XXX or chrX:XXX-XXX,strand (+/-/.)
        region_coords = region.split(",")
        # The default strand is ., which is neither strand
        strand = region_coords[1] if len(region_coords) > 1 else "."
        chrom, coords = region_coords[0].split(":")
        start, end = map(int, coords.split("-"))
        if window_size is None:
            regions_dict[chrom].append((start, end, strand))
        else:
            center_coord = (start + end) // 2
            regions_dict[chrom].append(
                (center_coord - window_size, center_coord + window_size, strand)
            )
    else:
        raise ValueError(
            f"Invalid regions {type(region)}: {region}. Please use the format chrX:XXX-XXX,strand."
        )


def bed_from_regions_dict(
    regions_dict: dict,
    save_bed_path: Path,
):
    with open(save_bed_path, "w") as processed_bed:
        for chrom, regions_list in regions_dict.items():
            for start, end, _ in regions_list:
                bed_line = (
                    "\t".join([chrom, str(start), str(end), ".", ".", "."]) + "\n"
                )
                processed_bed.write(bed_line)


def bedmethyl_to_bigwig(input_bedmethyl: str | Path, output_bigwig: str | Path):
    return 0


def check_len_equal(*args: list) -> bool:
    """
    Checks whether all provided lists are the same length.
    """
    return all(len(x) == len(args[0]) for x in args)


def bar_plot(categories: list[str], values: np.ndarray, y_label: str, **kwargs) -> Axes:
    """
    Utility for producing bar plots.

    Args:
        categories: parallel with values; bar labels
        values: parallel with categories: bar heights
        y_label: y-axis label
        kwargs: other keyword parameters passed through to seaborn.barplot

    Returns:
        Axes object containing the plot
    """
    axes = sns.barplot(x=categories, y=values, hue=categories, **kwargs)
    axes.set(ylabel=y_label)
    return axes


def line_plot(
    indep_vector: np.ndarray,
    indep_name: str,
    dep_vectors: list[np.ndarray],
    dep_names: list[str],
    y_label: str,
    **kwargs,
) -> Axes:
    """
    Utility for producing overlayed line plots for data vectors with the same x-axis values.

    Takes in one independent vector and arbitrarily many dependent vectors. Plots all dependent vectors on the same axes against the same dependent vector.
    All vectors must be of equal length.

    TODO: Right now, this always generates a legend with the title "variable". I could add a parameter to specify this (by passing the var_name argument to pd.DataFrame.melt), but then that percolates upwards to other methods. How to do this cleanly?

    Args:
        indep_vector: parallel with each entry in vectors; independent variable values shared across each overlayed line
        indep_name: name of independent variable; set as x axis label
        dep_vectors: outer list parallel with dep_names; each inner vector parallel with indep_vector; dependent variable values for each overlayed line
        dep_names: parallel with dep_vectors; names of each overlayed line; set as legend entries
        y_label: y-axis label
        kwargs: other keyword parameters passed through to seaborn.lineplot

    Returns:
        Axes object containing the plot

    Raises:
        ValueError: raised if any vectors are of unequal length
    """
    # construct dict of {vector_name: vector}, including the x vector using dict union operations
    data_dict = {indep_name: indep_vector} | dict(zip(dep_names, dep_vectors))
    # construct long-form data table for plotting
    try:
        data_table = pd.DataFrame(data_dict).melt(
            id_vars=indep_name, value_name=y_label
        )
    except ValueError as e:
        raise ValueError(
            "All dependent and independent vectors must be the same length"
        ) from e
    # plot lines
    return sns.lineplot(
        data=data_table, x=indep_name, y=y_label, hue="variable", **kwargs
    )


def smooth_rolling_mean(
    vector: np.ndarray[float], window: int, min_periods: int = 1
) -> np.ndarray:
    """
    Smooths the given vector, using rolling centered windows of the given size.
    See pandas rolling documentation for details; documentation for relevant arguments copied here.

    Note: Because this operation is always centered, min_periods only has an effect if it is less than half of window size.

    TODO: Is pandas the most efficient implementation for this?
    TODO: Is it reasonable for min_periods to be default 1? That makes some sense for plotting, but might make analysis misleading in the future, compared to defaulting to window size.

    Args:
        vector: the vector of values to smooth
        window: size of the moving window
        min_periods: minimum number of observations in window to output a value; otherwise, result is np.nan

    Returns:
        Vector of smoothed values
    """
    return (
        pd.Series(vector)
        .rolling(window=window, min_periods=min_periods, center=True)
        .mean()
        .values
    )
