from pathlib import Path

from dimelo.test import RelativePath

# Base input and output directories
test_data_dir = Path("./data")
output_dir = test_data_dir / "test_targets"

region = "chr1:114357437-114359753"  # 'chr1:9167177-9169177'

# Paths to input files
ctcf_bam_file = test_data_dir / "ctcf_demo.sorted.bam"
# ctcf_guppy_bam_file = test_data_dir / 'winnowmap_guppy_merge_subset.updated.bam'
ctcf_target_regions = RelativePath(test_data_dir / "ctcf_demo_peak.bed")
ctcf_off_target_regions = RelativePath(test_data_dir / "ctcf_demo_not_peak.bed")

ctcf_bam_file_updated = RelativePath("./output/ctcf_demo.updated.bam")
output_dir = RelativePath(output_dir)

test_matrix = {
    "megalodon_peaks_190": (
        # input kwargs
        {
            "input_file": ctcf_bam_file_updated,
            "output_name": "megalodon_peaks_190",
            "output_directory": output_dir,
            "regions": [ctcf_target_regions, ctcf_off_target_regions],
            "motifs": ["A,0", "CG,0"],
            "thresh": 190,
            "window_size": 5000,
            "sort_by": ["read_start", "read_name", "motif"],
            "smooth_window": 1,
            "title": "megalodon_peaks_190",
            "single_strand": False,
            "regions_5to3prime": False,
        },
        # outputs dict function:values
        {},  # populated in subsequent cells
    ),
    "megalodon_single_190": (
        # input kwargs
        {
            "input_file": ctcf_bam_file_updated,
            "output_name": "megalodon_single_190",
            "output_directory": output_dir,
            "regions": region,
            "motifs": ["A,0", "CG,0"],
            "thresh": 190,
            "window_size": None,
            "sort_by": ["read_start", "read_name", "motif"],
            "smooth_window": 10,
            "title": "megalodon_single_190",
            "single_strand": False,
            "regions_5to3prime": False,
        },
        # outputs dict function:values
        {},  # populated in subsequent cells
    ),
    "megalodon_single_and_peaks_190": (
        # input kwargs
        {
            "input_file": ctcf_bam_file_updated,
            "output_name": "megalodon_single_and_peaks_190",
            "output_directory": output_dir,
            "regions": [region, ctcf_target_regions, ctcf_off_target_regions],
            "motifs": ["A,0", "CG,0"],
            "thresh": 190,
            "window_size": 5000,
            "sort_by": ["read_start", "read_name", "motif"],
            "smooth_window": 100,
            "title": "megalodon_single_and_peaks_190",
            "single_strand": True,
            "regions_5to3prime": True,
        },
        # outputs dict function:values
        {},  # populated in subsequent cells
    ),
    "megalodon_peaks_nothresh": (
        # input kwargs
        {
            "input_file": ctcf_bam_file_updated,
            "output_name": "megalodon_peaks_nothresh",
            "output_directory": output_dir,
            "regions": [ctcf_target_regions, ctcf_off_target_regions],
            "motifs": ["A,0", "CG,0"],
            "thresh": None,
            "window_size": 5000,
            "sort_by": ["read_start", "read_name", "motif"],
            "smooth_window": 100,
            "title": "megalodon_peaks_nothresh",
            "single_strand": True,
            "regions_5to3prime": False,
        },
        # outputs dict function:values
        {},  # populated in subsequent cells
    ),
    "megalodon_single_nothresh": (
        # input kwargs
        {
            "input_file": ctcf_bam_file_updated,
            "output_name": "megalodon_single_nothresh",
            "output_directory": output_dir,
            "regions": region,
            "motifs": ["A,0", "CG,0"],
            "thresh": None,
            "window_size": 5000,
            "sort_by": ["read_start", "read_name", "motif"],
            "smooth_window": 1,
            "title": "megalodon_single_nothresh",
            "single_strand": False,
            "regions_5to3prime": True,
        },
        # outputs dict function:values
        {},  # populated in subsequent cells
    ),
}
