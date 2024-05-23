from pathlib import Path

import numpy as np
import pandas as pd
import plotly

from . import load_processed

# TODO: add plotly to requirements, probably?
# TODO: Need to add kaleido for image export: pip install kaleido
# TODO: Should this be allowed to take in mutliple regions? I think no.
# TODO: Should this method do saving, or just return a plotly object?

DEFAULT_THRESH_A = 129
DEFAULT_THRESH_C = 129
DEFAULT_DOTSIZE = 4
COLOR_A = "#053C5E"
COLOR_C = "#BB4430"


def plot_read_browser(
    mod_file_name: str | Path,
    # regions: str | Path | list[str | Path],
    region: str | Path,
    motifs: list[str],
    output_dir: str | Path,
    interactive_output: bool = True,
    static_formats: list[str] | None = None,
    static_width: int = 1000,
    static_height: int = 400,
    # window_size: int | None = None,
    # single_strand: bool = False,
    sort_by: str | list[str] = "shuffle",
    **kwargs,
) -> None:
    # TODO: is it actually necessary for this to be coerced here? Or is it okay to pass it down?
    mod_file_name = Path(mod_file_name)
    output_dir = Path(output_dir)

    # TODO: Should this be allowed to pass down the window_size and single_strand options? Do those make sense for a browser?
    read_tuples, entry_labels, _ = load_processed.read_vectors_from_hdf5(
        file=mod_file_name,
        regions=region,
        motifs=motifs,
        sort_by=sort_by,
        calculate_mod_fractions=False,
    )

    # Coerce read tuples to initial dataframe, throwing away unnecessary columns
    to_exclude = ["chromosome", "strand", "region_start", "region_end"]
    read_df = pd.DataFrame.from_records(
        read_tuples, columns=entry_labels, exclude=to_exclude
    )

    # For each row, pull out just the positions of valid bases and the probabilities at those positions
    # TODO: I don't like using iterrows, but it seems silly to have two calls to apply that do basically the same thing redundantly; just looping once instead
    prob_vectors = []
    pos_vectors = []
    for _, row in read_df.iterrows():
        selector = row.val_vector == 1
        all_positions = np.arange(len(row.mod_vector)) + row.read_start
        prob_vectors.append(row.mod_vector[selector])
        pos_vectors.append(all_positions[selector])
    read_df["prob_vector"] = prob_vectors
    read_df["pos_vector"] = pos_vectors

    # assign reads y-axis values based on their unique names, preserving the pre-set order
    # TODO: I'm pretty sure that however I do this, there will possibly be an "overdrawing" problem if the same read shows up multiple times. Do I care? Will it be taken care of by dropping duplicates?
    # TODO: There is probably a better way to do this. I investigated pd.Categorical, but couldn't figure out how to tell it to do it in the passed order.
    # TODO: This downcasting warning is INCREDIBLY vague, and I can't tell if it's actually a problem in this case. I even tried following their instructions and it didn't fix anything. UGHHHHHHHHHH...
    read_df["y_index"] = read_df.read_name.replace(
        {k: v for v, k in enumerate(read_df.read_name.unique())}
    )

    # TODO: Dropping duplicates should hide the "overdrawing" problem when reads are duplicated in the dataset. Is this a problem or the intended behavior?
    # Get two separate dataframes:
    # * represents the read extents, to draw the grey lines
    read_extent_df = read_df[["read_start", "read_end", "y_index"]].drop_duplicates()
    # * represents the methylation events
    mod_event_df = (
        read_df[["y_index", "motif", "pos_vector", "prob_vector"]]
        .explode(["pos_vector", "prob_vector"])
        .rename(columns={"pos_vector": "pos", "prob_vector": "prob"})
    )

    # TODO: I know there's an off by a little bit error here, but I don't care to fix it yet
    # Apply threshold to mod_event_df
    # mod_event_df.drop(mod_eve)
    thresh_float = DEFAULT_THRESH_A / 255
    mod_event_df = mod_event_df[mod_event_df.prob > thresh_float]

    # TODO: I feel like there has to be a cleaner way to do this, maybe using plotly express, but I dont know and I'm just trying to get this done first. Lots of iterrows. Sad.
    fig = plotly.graph_objects.Figure()
    for _, row in read_extent_df.iterrows():
        fig.add_trace(
            plotly.graph_objects.Scatter(
                x=[row.read_start, row.read_end],
                y=[row.y_index, row.y_index],
                mode="lines",
                line=dict(width=1, color="lightgrey"),
                showlegend=False,
            )
        )
    # TODO: This way of specifying color is very dumb.
    color_specs = [["white", COLOR_A], ["white", COLOR_C]]
    for motif_idx, (motif, motif_df) in enumerate(mod_event_df.groupby("motif")):
        min_overall = motif_df["prob"].min()
        max_overall = motif_df["prob"].max()
        fig.add_trace(
            plotly.graph_objs.Scatter(
                x=motif_df["pos"],
                y=motif_df["y_index"],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=DEFAULT_DOTSIZE,
                    color=motif_df["prob"],
                    # TODO: This way of specifying color is very dumb.
                    colorscale=color_specs[motif_idx],
                    colorbar=dict(
                        title=f"{motif} probability",
                        titleside="right",
                        tickmode="array",
                        tickvals=[min_overall, max_overall],
                        ticktext=[
                            str(round(min_overall, 2)),
                            str(round(max_overall, 2)),
                        ],
                        ticks="outside",
                        thickness=15,
                        # TODO: Is this positioning system dumb?
                        x=1 + (motif_idx * 0.10),
                    ),
                ),
            )
        )
    fig.update_layout(
        barmode="overlay",
        title="test_title",
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # TODO: If this is going to write things, it really needs to have a better file naming scheme...
    # Write static images
    for fmt in static_formats:
        fig.write_image(
            output_dir / f"read_browser.{fmt}", width=static_width, height=static_height
        )

    # Write html
    if interactive_output:
        fig.write_html(output_dir / "read_browser.html", include_plotlyjs="cdn")
