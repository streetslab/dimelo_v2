import gzip
import multiprocessing
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pysam
from tqdm.auto import tqdm

from . import run_modkit, utils

"""
This module contains code to convert .bam files into both human-readable and 
indexed random-access pileup and read-wise processed outputs.
"""

"""
Global variables
"""

# This should be updated in tandem with the environment.yml nanoporetech::modkit version
EXPECTED_MODKIT_VERSION = "0.2.4"

# Specifies how many reads to check for the base modifications of interest.
NUM_READS_TO_CHECK = 100


"""
Import checks
"""
# Add conda env bin folder to path if it is not already present
current_interpreter = sys.executable
env_bin_path = os.path.dirname(current_interpreter)
if env_bin_path not in os.environ["PATH"]:
    print(
        f"PATH does not include the conda environment /bin folder. Adding {env_bin_path}."
    )
    os.environ["PATH"] = f'{env_bin_path}:{os.environ["PATH"]}'
    print(f'PATH is now {os.environ["PATH"]}')

# Check modkit on first import
try:
    result = subprocess.run(["modkit", "--version"], stdout=subprocess.PIPE, text=True)
    modkit_version = result.stdout
    if modkit_version.split()[1] == EXPECTED_MODKIT_VERSION:
        print(f"modkit found with expected version {EXPECTED_MODKIT_VERSION}")
    else:
        print(
            f"modkit found with unexpected version {modkit_version.split()[1]}. Versions other than {EXPECTED_MODKIT_VERSION} may exhibit unexpected behavior. It is recommended that you use v{EXPECTED_MODKIT_VERSION}"
        )
except subprocess.CalledProcessError:
    print(
        'Executable not found for modkit. Install dimelo using "conda env create -f environment.yml" or install modkit manually to your conda environment using "conda install nanoporetech::modkit==0.2.4". Without modkit you cannot run parse_bam functions.'
    )

"""
User-facing parse operations: pileup and extract
"""


def pileup(
    input_file: str | Path,
    output_name: str,
    ref_genome: str | Path,
    output_directory: str | Path | None = None,
    regions: str | Path | list[str | Path] | None = None,
    motifs: list = ["A,0", "CG,0"],
    thresh: float | None = None,
    window_size: int | None = None,
    cores: int | None = None,
    log: bool = False,
    cleanup: bool = True,
    quiet: bool = False,
    override_checks: bool = False,
) -> tuple[Path, Path]:
    """
    TODO: Merge bed_file / region_str / window_size handling into a unified function somewhere

    Takes a file containing long read sequencing data aligned
    to a reference genome with modification calls for one or more base/context
    and creates a pileup. A pileup is a genome-position-wise sum of both reads with
    bases that could have the modification in question and of reads that are in
    fact modified.

    The current implementation of this method uses modkit, a tool built by
    Nanopore Technologies, along with htslib tools compress and index the output
    bedmethyl file.

    https://github.com/nanoporetech/modkit/

    Args:
        output_file: a string or Path object pointing to the location of a .bam file.
            The file should follow at least v1.6 of the .bam file specifications,
            found here: https://samtools.github.io/hts-specs/
            https://samtools.github.io/hts-specs/SAMv1.pdf

            The file needs to have modifications stored in the standard format,
            with MM and ML tags (NOT mm and ml) and mod names m for 5mC and a
            for 6mA.

            Furthermore, the file must have a .bam.bai index file with the same name.
            You can create an index if needed using samtools index.
        output_name: a string that will be used to create an output folder
            containing the intermediate and final outputs, along with any logs.
        ref_genome: a string of Path objecting pointing to the .fasta file
            for the reference genome to which the .bam file is aligned.
        output_directory: optional str or Path pointing to an output directory.
            If left as None, outputs will be stored in a new folder within the input
            directory.
        regions: TODO
        motifs: a list of strings specifying which base modifications to look for.
            The basemods are each specified as {sequence_motif},{position_of_modification}.
            For example, a methylated adenine is specified as 'A,0' and CpG methylation
            is specified as 'CG,0'.
        thresh: float point number specifying the base modification probability threshold
            used to delineate modificaton calls as True or False. When set to None, modkit
            will select its own threshold automatically based on the data.
        window_size: an integer specifying a window around the center of each bed_file
            region. If set to None, the bed_file is used unmodified. If set to a non-zero
            positive integer, the bed_file regions are replaced by new regions with that
            window size in either direction of the center of the original bed_file regions.
            This is used for e.g. extracting information from around known motifs or peaks.
        cores: an integer specifying how many parallel cores modkit gets to use.
            By default modkit will use all of the available cores on the machine.
        log: a boolean specifying whether to output logs into the output folder.
        cleanup: a boolean specifying whether to clean up to keep intermediate
            outputs. The final processed files are not human-readable, whereas the intermediate
            outputs are. However, intermediate outputs can also be quite large.
        override_checks: convert errors from input checking into warnings if True

    Returns:
        Path object pointing to the compressed and indexed .bed.gz bedmethyl file, ready
        for plotting functions.
        Path object pointing to regions.processed.bed

    """
    """
    TODO: There are a lot of issues that are all related here:
    dimelo/parse_bam.py:150: error: Incompatible types in assignment (expression has type "Path | None", variable has type "str | Path")  [assignment]
    dimelo/parse_bam.py:169: error: Argument "input_file" to "prep_outputs" has incompatible type "str | Path"; expected "Path"  [arg-type]
    dimelo/parse_bam.py:256: error: Argument "input_file" to "run_with_progress_bars" has incompatible type "str | Path"; expected "Path"  [arg-type]
    dimelo/parse_bam.py:257: error: Argument "ref_genome" to "run_with_progress_bars" has incompatible type "str | Path"; expected "Path"  [arg-type]
    
    I'm not sure of the most elegant way to fix it. Come back and address.
    """
    input_file, ref_genome, output_directory = sanitize_path_args(
        input_file, ref_genome, output_directory
    )

    try:
        verify_inputs(input_file, motifs, ref_genome)
    except Exception as e:
        if override_checks:
            if not quiet:
                print(f"WARNING: {e}")
        else:
            raise Exception(
                f'{e}\nIf you are confident that your inputs are ok, pass "override_checks=True" to convert to warning and proceed with processing.'
            ) from e

    output_path, (output_bedmethyl, output_bedmethyl_sorted, output_bedgz_sorted, _) = (
        prep_outputs(
            output_directory=output_directory,
            output_name=output_name,
            input_file=input_file,
            output_file_names=[
                "pileup.bed",
                "pileup.sorted.bed",
                "pileup.sorted.bed.gz",
                "pileup.sorted.bed.gz.tbi",
            ],
        )
    )

    # TODO: This is mildly confusing. I get what it's doing, but it's hard to follow / names are bad. Also, why is it used in cleanup here, but not in extract?
    region_specifier, bed_filepath_processed = create_region_specifier(
        output_path,
        regions,
        window_size,
    )

    motif_command_list = []
    if len(motifs) > 0:
        for motif in motifs:
            parsed_motif = utils.ParsedMotif(motif)
            motif_command_present = False
            for a, b in zip(motif_command_list, motif_command_list[1:]):
                if a == parsed_motif.motif_seq and b == str(parsed_motif.modified_pos):
                    # This motif is already going to be processed; we want to skip adding it a second
                    # time because modkit does not like duplicate motifs.
                    # It's actually ok if it's a different mod code in the two cases because the pileup
                    # operation, under the hood, keeps all mod codes. Filtering is only done when loading.
                    motif_command_present = True
                    break
            if not motif_command_present:
                motif_command_list.append("--motif")
                motif_command_list.append(parsed_motif.motif_seq)
                motif_command_list.append(str(parsed_motif.modified_pos))
    else:
        raise ValueError("Error: no motifs specified. Nothing to process.")

    if log:
        if not quiet:
            print("Logging to ", Path(output_path) / "pileup-log")
        log_command = ["--log-filepath", Path(output_path) / "pileup-log"]
    else:
        log_command = []

    # TODO: This should be a method, like create_region_specifier, or just combined into a prep method for the start...
    cores_avail = multiprocessing.cpu_count()
    if cores is None:
        if not quiet:
            print(
                f"No specified number of cores requested. {cores_avail} available on machine, allocating all."
            )
        cores_command_list = ["--threads", str(cores_avail)]
    elif cores > cores_avail:
        if not quiet:
            print(
                f"Warning: {cores} cores request, {cores_avail} available. Allocating {cores_avail}"
            )
        cores_command_list = ["--threads", str(cores_avail)]
    else:
        if not quiet:
            print(f"Allocating requested {cores} cores.")
        cores_command_list = ["--threads", str(cores)]

    # TODO: This is SO SO SO similar to extract; just the ValueError vs. printing. I think this can be resolved
    mod_thresh_list: list[str] = []
    if thresh is None:
        if not quiet:
            print(
                "No base modification threshold provided. Using adaptive threshold selection via modkit."
            )
    else:
        adjusted_threshold = utils.adjust_threshold(thresh, quiet=quiet)
        if adjusted_threshold < 0.5 and not quiet:
            print(
                f"WARNING: thresh {thresh} is very low and may lead to unexpected behavior. Typical thresholds are at least 0.5 or 128."
            )
        for motif in motifs:
            parsed_motif = utils.ParsedMotif(motif)
            for mod_code in parsed_motif.mod_codes:
                mod_thresh_list = mod_thresh_list + [
                    "--mod-thresholds",
                    f"{mod_code}:{adjusted_threshold}",
                ]

    pileup_command_list = (
        ["modkit", "pileup", input_file, output_bedmethyl]
        + region_specifier
        + motif_command_list
        + ["--ref", ref_genome, "--filter-threshold", "0"]
        + mod_thresh_list
        + cores_command_list
        + log_command
    )

    # TODO: Do we need to store and use the output from this method? Previously was being printed immediately afterward.
    _ = run_modkit.run_with_progress_bars(
        command_list=pileup_command_list,
        input_file=input_file,
        ref_genome=ref_genome,
        motifs=motifs,
        load_fasta_regex=r"\s+\[.*?\]\s+(\d+)\s+Reading",
        find_motifs_regex=r"\s+(\d+)/(\d+)\s+finding\s+([A-Za-z0-9,]+)\s+motifs",
        contigs_progress_regex=r"\s+(\d+)/(\d+)\s+contigs",
        single_contig_regex=r"\s+(\d+)/(\d+)\s+processing\s+([\w]+)[^\w]",
        buffer_size=50,
        progress_granularity=25,
        done_str="Done",
        err_str="Error",
        expect_done=True,
        quiet=quiet,
    )
    # print(done_string)

    with open(output_bedmethyl_sorted, "w") as sorted_file:
        subprocess.run(
            ["sort", "-k1,1", "-k2,2n", output_bedmethyl], stdout=sorted_file
        )
    pysam.tabix_compress(output_bedmethyl_sorted, output_bedgz_sorted, force=True)
    pysam.tabix_index(str(output_bedgz_sorted), preset="bed", force=True)

    # TODO: Can cleanup be consolidated?
    if cleanup:
        if output_bedmethyl.exists():
            output_bedmethyl.unlink()
        if output_bedmethyl_sorted.exists():
            output_bedmethyl_sorted.unlink()

    return output_bedgz_sorted, bed_filepath_processed


def extract(
    input_file: str | Path,
    output_name: str,
    ref_genome: str | Path,
    output_directory: str | Path | None = None,
    regions: str | Path | list[str | Path] | None = None,
    motifs: list = ["A,0", "CG,0", "GCH,1"],
    thresh: float | None = None,
    window_size: int | None = None,
    cores: int | None = None,
    log: bool = False,
    cleanup: bool = True,
    quiet: bool = False,
    override_checks: bool = False,
) -> tuple[Path, Path]:
    """
    TODO: Merge bed_file / region_str / window_size handling into a unified function somewhere

    Takes a file containing long read sequencing data aligned
    to a reference genome with modification calls for one or more base/context
    and pulls out data from each individual read. The intermediate outputs contain
    a plain-text list of all base modifications, split out by type. The compressed
    and indexed output contains vectors of valid and modified positions within each
    read.

    The current implementation of this method uses modkit, a tool built by
    Nanopore Technologies, along with h5py to build the final output file.

    https://github.com/nanoporetech/modkit/

    Args:
        output_file: a string or Path object pointing to the location of a .bam file.
            The file should follow at least v1.6 of the .bam file specifications,
            found here: https://samtools.github.io/hts-specs/
            https://samtools.github.io/hts-specs/SAMv1.pdf

            The file needs to have modifications stored in the standard format,
            with MM and ML tags (NOT mm and ml) and mod names m for 5mC and a
            for 6mA.

            Furthermore, the file must have a .bam.bai index file with the same name.
            You can create an index if needed using samtools index.
        output_name: a string that will be used to create an output folder
            containing the intermediate and final outputs, along with any logs.
        ref_genome: a string of Path objecting pointing to the .fasta file
            for the reference genome to which the .bam file is aligned.
        output_directory: optional str or Path pointing to an output directory.
            If left as None, outputs will be stored in a new folder within the input
            directory.
        regions: TODO
        motifs: a list of strings specifying which base modifications to look for.
            The basemods are each specified as {sequence_motif},{position_of_modification}.
            For example, a methylated adenine is specified as 'A,0' and CpG methylation
            is specified as 'CG,0'.
        thresh: float point number specifying the base modification probability threshold
            used to delineate modificaton calls as True or False. When set to None, modkit
            will select its own threshold automatically based on the data.
        window_size: an integer specifying a window around the center of each bed_file
            region. If set to None, the bed_file is used unmodified. If set to a non-zero
            positive integer, the bed_file regions are replaced by new regions with that
            window size in either direction of the center of the original bed_file regions.
            This is used for e.g. extracting information from around known motifs or peaks.
        cores: an integer specifying how many parallel cores modkit gets to use.
            By default modkit will use all of the available cores on the machine.
        log: a boolean specifying whether to output logs into the output folder.
        cleanup: a boolean specifying whether to clean up to keep intermediate
            outputs. The final processed files are not human-readable, whereas the intermediate
            outputs are. However, intermediate outputs can also be quite large.
        override_checks: convert errors from input checking into warnings if True

    Returns:
        Path object pointing to the compressed and indexed output .h5 file, ready for
        plotting functions.
        Path object pointing to regions.processed.bed

    """
    """
    TODO: There are a lot of issues that are all related here:
    dimelo/parse_bam.py:374: error: Incompatible types in assignment (expression has type "Path | None", variable has type "str | Path")  [assignment]
    dimelo/parse_bam.py:393: error: Argument "input_file" to "prep_outputs" has incompatible type "str | Path"; expected "Path"  [arg-type]
    dimelo/parse_bam.py:480: error: Argument "input_file" to "run_with_progress_bars" has incompatible type "str | Path"; expected "Path"  [arg-type]
    dimelo/parse_bam.py:481: error: Argument "ref_genome" to "run_with_progress_bars" has incompatible type "str | Path"; expected "Path"  [arg-type]
    
    I'm not sure of the most elegant way to fix it. Come back and address.
    """
    input_file, ref_genome, output_directory = sanitize_path_args(
        input_file, ref_genome, output_directory
    )

    try:
        verify_inputs(input_file, motifs, ref_genome)
    except Exception as e:
        if override_checks:
            if not quiet:
                print(f"WARNING: {e}")
        else:
            raise Exception(
                f'{e}\nIf you are confident that your inputs are ok, pass "override_checks=True" to convert to warning and proceed with processing.'
            ) from e

    # TODO: Add intermediate mod-specific .txt files?
    output_path, (output_h5,) = prep_outputs(
        output_directory=output_directory,
        output_name=output_name,
        input_file=input_file,
        output_file_names=["reads.combined_basemods.h5"],
    )

    region_specifier, bed_filepath_processed = create_region_specifier(
        output_path,
        regions,
        window_size,
    )

    cores_avail = multiprocessing.cpu_count()
    if cores is None:
        if not quiet:
            print(
                f"No specified number of cores requested. {cores_avail} available on machine, allocating all."
            )
        cores_command_list = ["--threads", str(cores_avail)]
    elif cores > cores_avail:
        if not quiet:
            print(
                f"Warning: {cores} cores request, {cores_avail} available. Allocating {cores_avail}"
            )
        cores_command_list = ["--threads", str(cores_avail)]
    else:
        if not quiet:
            print(f"Allocating requested {cores} cores.")
        cores_command_list = ["--threads", str(cores)]

    mod_thresh_list: list[str] = []
    if thresh is None:
        if not quiet:
            print(
                "No valid base modification threshold provided. Raw probs will be saved."
            )
        adjusted_threshold = None
    else:
        adjusted_threshold = utils.adjust_threshold(thresh, quiet=quiet)
        if adjusted_threshold < 0.5 and not quiet:
            print(
                f"WARNING: thresh {thresh} is very low and may lead to unexpected behavior. Typical thresholds are at least 0.5 or 128."
            )
        for motif in motifs:
            parsed_motif = utils.ParsedMotif(motif)
            for mod_code in parsed_motif.mod_codes:
                mod_thresh_list = mod_thresh_list + [
                    "--mod-thresholds",
                    f"{mod_code}:{adjusted_threshold}",
                ]

    if log:
        if not quiet:
            print("logging to ", Path(output_path) / "extract-log")
        log_command = ["--log-filepath", Path(output_path) / "extract-log"]
    else:
        log_command = []

    for motif in motifs:
        # print(f'Extracting {basemod} sites')
        motif_command_list = []
        parsed_motif = utils.ParsedMotif(motif)
        motif_command_list.append("--motif")
        motif_command_list.append(parsed_motif.motif_seq)
        motif_command_list.append(str(parsed_motif.modified_pos))

        output_txt = Path(output_path) / (f"reads.{motif}.txt")

        if os.path.exists(output_txt):
            os.remove(output_txt)

        extract_command_list = (
            ["modkit", "extract", input_file, output_txt]
            + region_specifier
            + motif_command_list
            + cores_command_list
            + log_command
            + [
                "--ref",
                ref_genome,
                "--filter-threshold",
                "0",
            ]
        )

        # TODO: Do we need to store and use the output from this method? Previously was being printed immediately afterward.
        # This is something the user might want to see - it's the end-of-process message for modkit, says e.g. how many reads were processed and stuff
        _ = run_modkit.run_with_progress_bars(
            command_list=extract_command_list,
            input_file=input_file,
            ref_genome=ref_genome,
            motifs=[motif],
            load_fasta_regex=r"\s+\[.*?\]\s+(\d+)\s+parsing FASTA",
            find_motifs_regex=r"\s+(\d+)/(\d+)\s+([\w]+)\s+searched",
            contigs_progress_regex=r"\s+(\d+)/(\d+)\s+contigs\s+[^s]",
            single_contig_regex=r"\s+(\d+)/(\d+)\s+processing\s+([\w]+)[^\w]",
            buffer_size=100,
            progress_granularity=50,
            done_str="Done",
            err_str="Error",
            expect_done=False,
            quiet=quiet,
        )
        # print(done_string)

        # print(f'Adding {basemod} to {output_h5}')
        read_by_base_txt_to_hdf5(
            output_txt,
            output_h5,
            motif,
            adjusted_threshold,
            quiet=quiet,
        )
        if cleanup:
            os.remove(output_txt)

    return output_h5, bed_filepath_processed


"""
Helper functions to facilitate bam parse operations

check_bam_format: verify that a bam is formatted correctly to be processed.
create_region_specifier: create a list to append to the modkit call for specifying genomic regions.
adjust_threshold: backwards-compatible threshold adjustment, i.e. taking 0-255 thresholds and turning
    them into 0-1.
read_by_base_txt_to_hdf5: convert modkit extract txt into an .h5 file for rapid read access.
"""


def verify_inputs(
    input_file,
    motifs,
    ref_genome,
):
    check_bam_format(input_file, motifs)
    correct_bases, total_bases = get_alignment_quality(input_file, ref_genome)
    if total_bases == 0:
        raise ValueError(
            f"First {NUM_READS_TO_CHECK} reads are empty. Please verify your {input_file.name} contents."
        )
    elif correct_bases / total_bases < 0.35:
        raise ValueError(
            f"First {NUM_READS_TO_CHECK} reads have anomalously low alignment quality: only {100*correct_bases/total_bases}% of bases align.\nPlease verify that {input_file.name} is actually aligned to {ref_genome.name}."
        )
    return


def check_bam_format(
    bam_file: str | Path,
    motifs: list = ["A,0", "CG,0"],
):
    """
    Check whether a .bam file is formatted appropriately for modkit

    Args:
        bam_file: a formatted .bam file with a .bai index
        basemods: a list of base modification motifs

    Returns:
        None. If the function returns, you are ok.

    """
    basemods_found_dict = {}
    mod_codes_dict = {}
    mod_codes_found_dict = defaultdict(set)
    for motif in motifs:
        parsed_motif = utils.ParsedMotif(motif)
        mod_codes_dict[parsed_motif.modified_base] = parsed_motif.mod_codes
        basemods_found_dict[parsed_motif.modified_base] = False

    input_bam = pysam.AlignmentFile(bam_file)

    try:
        for counter, read in enumerate(input_bam.fetch()):
            read_dict = read.to_dict()
            for tag_string in read_dict["tags"]:
                tag = tag_string.split(",")[0].split(":")[0]
                if tag == "Mm" or tag == "Ml":
                    raise ValueError(
                        f'Base modification tags are out of spec (Mm and Ml instead of MM and ML). \n\nConsider using "modkit update-tags {str(bam_file)} new_file.bam" in the command line with your conda environment active and then trying with the new file. For megalodon basecalling/modcalling, you may also need to pass "--mode ambiguous.\nBe sure to index the resulting .bam file."'
                    )
                elif tag == "MM":
                    for tag_substring in tag_string.split(";"):
                        tag_fields = tag_substring.split(",")[0].split(":")
                        if len(tag_fields) >= 3:
                            tag_value = tag_fields[2]
                        else:
                            tag_value = tag_fields[0]
                        if (
                            len(tag_value) > 0
                            and tag_value[-1] != "?"
                            and tag_value[-1] != "."
                        ):
                            raise ValueError(
                                f'Base modification tags are out of spec. Need ? or . in TAG:TYPE:VALUE for MM tag, else modified probability is considered to be implicit. \n\nConsider using "modkit update-tags {str(bam_file)} new_file.bam --mode ambiguous" in the command line with your conda environment active and then trying with the new file.'
                            )
                        else:
                            if (
                                len(tag_value) > 0
                                and tag_value[0] in basemods_found_dict
                            ):
                                correct_mod_codes = mod_codes_dict[tag_value[0]]
                                # valid_mod_codes = mod_codes_dict[tag_value[0]].union(
                                #     utils.BASEMOD_NAMES_DICT[tag_value[0]]
                                # )
                                if tag_value[2] in correct_mod_codes:
                                    basemods_found_dict[tag_value[0]] = True
                                else:
                                    mod_codes_found_dict[tag_value[0]].add(tag_value[2])
                                # With the mode-code-aware motifs, it no longer makes sense to throw this error
                                # This is because the warning the user gets if their mod code isn't found, or (if none is specified)
                                # the default mod codes aren't found, can tell them what mod codes *were* found and they can add them
                                # to their motif or use adjust_mods according to what makes sense. Thus, unexpected codes are not a
                                # problem (in part this is because parse_bam will now set thresholds for the motif-specified OR default mod codes)
                                # elif tag_value[2] not in valid_mod_codes:
                                #     raise ValueError(
                                #         f'Base modification name unexpected: {tag_value[2]} to modify {tag_value[0]}, should be in set {valid_mod_codes}. \n\nIf you know what your mod names correspond to in terms of the latest .bam standard, consider using "modkit adjust-mods {str(bam_file)} new_file.bam --convert 5mC_name m --convert N6mA_name a --convert other_basemod_name correct_label" and then trying with the new file. Note: currently supported mod names are {utils.BASEMOD_NAMES_DICT}'
                                #     )
            if all(basemods_found_dict.values()):
                return
            if counter >= NUM_READS_TO_CHECK:
                missing_bases = []
                for base, found in basemods_found_dict.items():
                    if not found:
                        missing_bases.append(base)
                print(
                    f"""
WARNING: no modified appropriately-coded values found for {missing_bases} in the first {counter} reads. 
Do you expect this file to contain these modifications? parse_bam is looking for {motifs} but for {missing_bases} found only found {[f'{base}+{mod_codes}' for base, mod_codes in mod_codes_found_dict.items()]}.

Consider passing only the motifs and mod codes (e.g. m,h,a) that you expect to be present in your file. 
You can use modkit adjust-mods --convert <CONVERT> <CONVERT> [OPTIONS] <IN_BAM> <OUT_BAM> to update or consolidate mod codes.
See https://github.com/nanoporetech/modkit/blob/master/book/src/advanced_usage.md
                    """
                )
                return
    except ValueError as e:
        if "fetch called on bamfile without index" in str(e):
            raise ValueError(
                f'{e}. Consider using "samtools index {str(bam_file)}" to create an index if your .bam is already sorted.'
            ) from e
        else:
            raise
    except:
        raise


def get_alignment_quality(
    bam_file,
    ref_genome,
) -> tuple[int, int]:
    ref_genome_index = ref_genome.parent / (ref_genome.name + ".fai")
    if not ref_genome_index.exists():
        print(f"Indexing {ref_genome.name}. This only needs to be done once.")
        pysam.faidx(str(ref_genome))
    input_bam = pysam.AlignmentFile(bam_file, "rb")
    genome_fasta = pysam.FastaFile(str(ref_genome))
    total_bases = 0
    correct_bases = 0
    # For NUM_READS_TO_CHECK=100 this is <1s on most machines
    for index, read in enumerate(input_bam.fetch()):
        if index >= NUM_READS_TO_CHECK:
            return correct_bases, total_bases

        # The query sequence is the entire sequence as stored in the .bam file
        # So it is reverse complemented if it was a reverse read
        # Meaning we can compare it directly against the reference genome
        read_sequence = read.query_sequence

        # print(read.mapping_quality)

        # get_aligned_pairs returns a list of (read_coord,ref_coord) pairs with None values when not aligned
        # So if we just skip Nones and compare the remainder it'll tell us the accuracy

        for pos_in_read, pos_in_ref in read.get_aligned_pairs():
            if pos_in_read is not None and pos_in_ref is not None:
                total_bases += 1
                if read_sequence[pos_in_read] == str(
                    genome_fasta.fetch(read.reference_name, pos_in_ref, pos_in_ref + 1)
                ):
                    correct_bases += 1

    return correct_bases, total_bases


def create_region_specifier(
    output_path,
    regions,
    window_size,
):
    """
    Creates commands to pass to modkit based on bed_file regions.
    """

    if regions is not None:
        bed_filepath_processed = output_path / "regions.processed.bed"
        regions_dict = utils.regions_dict_from_input(
            regions,
            window_size,
        )
        utils.bed_from_regions_dict(regions_dict, bed_filepath_processed)
        region_specifier = ["--include-bed", str(bed_filepath_processed)]

    else:
        bed_filepath_processed = None
        region_specifier = []

    return region_specifier, bed_filepath_processed


def read_by_base_txt_to_hdf5(
    input_txt: str | Path,
    output_h5: str | Path,
    motif: str,
    thresh: float | None = None,
    quiet: bool = False,
    compress_level: int = 1,
    write_chunks: int = 1000,
) -> None:
    """
    Takes in a txt file generated by modkit extract and appends
    all the data from a specified basemod into an hdf5 file. If a thresh is specified, it
    also binarizes the mod calls.

    Args:
        input_txt: a string or Path pointing to a modkit extracted base-by-base modifications
            file. This file is assumed to have been created by modkit v0.2.4, other versions may
            have a different format and may not function normally.
        output_h5: a string or Path pointing to a valid place to save an .h5 file. If this
            file already exists, it will not be cleared and will simply be appended to. If it does
            not exist it will be created and datasets will be added for read_name, chromosome, read_start,
            read_end, base modification motif, mod_vector, and val_vector.
        basemod: a string specifying a single base modification. Basemods are specified as
            {sequence_motif},{position_of_modification}. For example, a methylated adenine is specified
            as 'A,0' and CpG methylation is specified as 'CG,0'.
        thresh: a floating point threshold for base modification calling, between zero and one.
            If specified as None, raw probabilities will be saved in the .h5 output.
        quiet: if True, this suppresses outputs
        compress_level: gzip compression level for datasets, specifically for vectors for now

    Returns:
        None

    """
    """
    TODO: There are some issues that are all related here:
    dimelo/parse_bam.py:718: error: Incompatible types in assignment (expression has type "Path | None", variable has type "str | Path")  [assignment]
    dimelo/parse_bam.py:725: error: Item "str" of "str | Path" has no attribute "open"  [union-attr]
    dimelo/parse_bam.py:890: error: Item "str" of "str | Path" has no attribute "name"  [union-attr]
    
    I'm not sure of the most elegant way to fix it. Come back and address.
    """
    input_txt, output_h5 = sanitize_path_args(input_txt, output_h5)

    parsed_motif = utils.ParsedMotif(motif)

    read_name = ""
    num_reads = 0
    # TODO: I think the function calls can be consolidated; lots of repetition
    with input_txt.open() as txt:
        for index, line in enumerate(txt):
            fields = line.split("\t")
            if index > 0 and read_name != fields[0]:
                read_name = fields[0]
                num_reads += 1
        num_lines = index
        # print(f'{num_reads} reads found in {input_txt}')
        txt.seek(0)
        with h5py.File(output_h5, "a") as h5:
            # Set dataset types

            # metadata strings
            dt_str = h5py.string_dtype(encoding="utf-8")
            # mod and val vectors -> uint8 allows us to just write whatever bytes we want
            # h5py does not appear to otherwise support vlen binary
            dt_vlen = h5py.vlen_dtype(np.dtype("uint8"))
            threshold_to_store = np.nan if thresh is None else thresh
            # Create a threshold dataset to store whether this data is thresholded (binary) or raw (float16)
            if "threshold" in h5:
                threshold_from_existing = h5["threshold"][()]
                if threshold_from_existing != threshold_to_store and not (
                    np.isnan(threshold_from_existing) and np.isnan(threshold_to_store)
                ):
                    raise ValueError(
                        "existing threshold in output_h5 does not match provided threshold for read_by_base_txt_to_hdf5."
                    )
            else:
                h5.create_dataset("threshold", data=threshold_to_store)
            # Create metadata datasets
            if "read_name" in h5:
                old_size = h5["read_name"].shape[0]
                h5["read_name"].resize((old_size + num_reads,))
            else:
                old_size = 0
                h5.create_dataset(
                    "read_name",
                    (num_reads,),
                    maxshape=(None,),
                    dtype=dt_str,
                    compression="gzip",
                    compression_opts=9,
                )
            if "chromosome" in h5:
                if old_size != h5["chromosome"].shape[0]:
                    print("size mismatch: read_name:chromosome")
                else:
                    h5["chromosome"].resize((old_size + num_reads,))
            else:
                h5.create_dataset(
                    "chromosome",
                    (num_reads,),
                    maxshape=(None,),
                    dtype=dt_str,
                    compression="gzip",
                    compression_opts=9,
                )
            if "read_start" in h5:
                if old_size != h5["read_start"].shape[0]:
                    print("size mismatch", "read_name", "read_start")
                else:
                    h5["read_start"].resize((old_size + num_reads,))
            else:
                h5.create_dataset(
                    "read_start",
                    (num_reads,),
                    maxshape=(None,),
                    dtype="i",
                    compression="gzip",
                    compression_opts=9,
                )
            if "read_end" in h5:
                if old_size != h5["read_end"].shape[0]:
                    print("size mismatch", "read_name", "read_end")
                else:
                    h5["read_end"].resize((old_size + num_reads,))
            else:
                h5.create_dataset(
                    "read_end",
                    (num_reads,),
                    maxshape=(None,),
                    dtype="i",
                    compression="gzip",
                    compression_opts=9,
                )
            if "strand" in h5:
                if old_size != h5["strand"].shape[0]:
                    print("size mismatch", "read_name", "strand")
                else:
                    h5["strand"].resize((old_size + num_reads,))
            else:
                h5.create_dataset(
                    "strand",
                    (num_reads,),
                    maxshape=(None,),
                    dtype=dt_str,
                    compression="gzip",
                    compression_opts=9,
                )
            if "motif" in h5:
                if old_size != h5["motif"].shape[0]:
                    print("size mismatch", "read_name", "motif")
                else:
                    h5["motif"].resize((old_size + num_reads,))
            else:
                h5.create_dataset(
                    "motif",
                    (num_reads,),
                    maxshape=(None,),
                    dtype=dt_str,
                    compression="gzip",
                    compression_opts=9,
                )
            # Create the vector datasets. These will contain raw bytes formatted into a uint8 array
            if "mod_vector" in h5:
                if old_size != h5["mod_vector"].shape[0]:
                    print("size mismatch read_name:mod_vector")
                else:
                    h5["mod_vector"].resize((old_size + num_reads,))
            else:
                h5.create_dataset(
                    "mod_vector",
                    (num_reads,),
                    maxshape=(None,),
                    dtype=dt_vlen,
                    # compression='gzip', # we are handling compression ourselves because hdf5 is bad at it
                    # compression_opts=9,
                )
            if "val_vector" in h5:
                if old_size != h5["val_vector"].shape[0]:
                    print("size mismatch read_name:val_vector")
                else:
                    h5["val_vector"].resize((old_size + num_reads,))
            else:
                h5.create_dataset(
                    "val_vector",
                    (num_reads,),
                    maxshape=(None,),
                    dtype=dt_vlen,
                    # compression='gzip', # we are handling compression ourselves because hdf5 is bad at it
                    # compression_opts=9,
                )

            #         next(txt)
            # Initialize loop vars
            read_name = ""
            read_chrom = ""
            read_len = 0
            ref_strand = ""
            read_start = 0
            read_end = 0
            val_coordinates_list: list[int] = []
            mod_values_list: list[float] = []

            read_counter = 0
            # TODO: This typing is correct, but maybe confusing? Is a defaultdict the best way to handle this, or should it be a more bespoke structure?
            read_dict_of_lists: defaultdict[str, list[str | int]] = defaultdict(list)
            reads_in_chunk = 0

            if quiet:
                iterator = enumerate(txt)
            else:
                iterator = tqdm(
                    enumerate(txt),
                    total=num_lines + 1,
                    desc=f"Transferring {num_reads} from {input_txt.name} into {output_h5.name}, new size {old_size+num_reads}",
                    bar_format="{bar}| {desc} {percentage:3.0f}% | {elapsed}<{remaining}",
                )
            for index, line in iterator:
                if index == 0:
                    #                     print(line)
                    continue
                fields = line.split("\t")
                pos_in_genome = int(fields[2])
                canonical_base = fields[15]
                prob = float(fields[10])
                mod_code = fields[11]

                if read_name != fields[0]:
                    # Record the read details unless this is the first read
                    if index > 1:
                        if len(val_coordinates_list) > 0:
                            read_len_along_ref = max(val_coordinates_list) + 1
                        else:
                            read_len_along_ref = read_len
                        mod_vector = np.zeros(read_len_along_ref, dtype=np.uint8)
                        if thresh is None:
                            # We subtract 0.25 because in modkit they add 0.5, but our elements are zero when the
                            # base motif isn't present, so to get things to round to the right integers to match the
                            # original .bam file, subtracting 0.25 is good. Anything from 0.001 to 0.4999 would work I think
                            mod_vector[val_coordinates_list] = np.rint(
                                np.array(mod_values_list) * 256 - 0.25
                            ).astype(np.uint8)
                        else:
                            mod_vector[val_coordinates_list] = np.array(
                                mod_values_list
                            ).astype(np.uint8)
                        val_vector = np.zeros(read_len_along_ref, dtype=np.uint8)
                        val_vector[val_coordinates_list] = 1
                        # Build mod_vector and val_vector from lists
                        read_dict_of_lists["read_name"].append(read_name)
                        read_dict_of_lists["chromosome"].append(read_chrom)
                        read_dict_of_lists["read_start"].append(read_start)
                        read_dict_of_lists["read_end"].append(read_end)
                        read_dict_of_lists["strand"].append(ref_strand)
                        read_dict_of_lists["motif"].append(motif)
                        read_dict_of_lists["mod_vector"].append(
                            np.frombuffer(
                                gzip.compress(
                                    mod_vector.tobytes(), compresslevel=compress_level
                                ),
                                dtype=np.uint8,
                            )
                        )
                        read_dict_of_lists["val_vector"].append(
                            np.frombuffer(
                                gzip.compress(
                                    val_vector.tobytes(), compresslevel=compress_level
                                ),
                                dtype=np.uint8,
                            )
                        )
                        reads_in_chunk += 1
                        if reads_in_chunk >= write_chunks:
                            for dataset, entry in read_dict_of_lists.items():
                                h5[dataset][
                                    old_size
                                    + (read_counter // write_chunks)
                                    * write_chunks : old_size + read_counter + 1
                                ] = entry
                            read_dict_of_lists = defaultdict(list)
                            reads_in_chunk = 0
                        read_counter += 1
                    # Set the read name of the next read
                    read_name = fields[0]
                    # Store some relevant read metadata
                    read_chrom = fields[3]
                    read_len = int(fields[9])
                    ref_strand = fields[5]
                    if ref_strand == "+":
                        pos_in_read_ref = int(fields[1])
                    elif ref_strand == "-":
                        pos_in_read_ref = read_len - int(fields[1]) - 1
                    # Calculate read info
                    read_start = pos_in_genome - pos_in_read_ref
                    read_end = read_start + read_len
                    # Instantiate lists
                    mod_values_list = []
                    val_coordinates_list = []

                # Regardless of whether its a new read or not,
                # add modification to vector if motif type is correct
                # for the motif in question
                if (
                    canonical_base == parsed_motif.modified_base
                    and mod_code in parsed_motif.mod_codes
                ):
                    val_coordinates_list.append(pos_in_genome - read_start)
                    if thresh is None:
                        mod_values_list.append(prob)
                    elif prob >= thresh:
                        mod_values_list.append(1)
                    else:
                        mod_values_list.append(0)

            # Save the last read
            if len(read_name) > 0:
                # Build the vectors
                if len(val_coordinates_list) > 0:
                    read_len_along_ref = max(val_coordinates_list) + 1
                else:
                    read_len_along_ref = read_len
                mod_vector = np.zeros(read_len_along_ref, dtype=np.uint8)
                if thresh is None:
                    # We subtract 0.25 because in modkit they add 0.5, but our elements are zero when the
                    # base motif isn't present, so to get things to round to the right integers to match the
                    # original .bam file, subtracting 0.25 is good. Anything from 0.001 to 0.4999 would work I think
                    mod_vector[val_coordinates_list] = np.rint(
                        np.array(mod_values_list) * 256 - 0.25
                    ).astype(np.uint8)
                else:
                    mod_vector[val_coordinates_list] = np.array(mod_values_list).astype(
                        np.uint8
                    )
                val_vector = np.zeros(read_len_along_ref, dtype=np.uint8)
                val_vector[val_coordinates_list] = 1
                val_vector = np.zeros(read_len_along_ref, dtype=np.uint8)
                val_vector[val_coordinates_list] = 1
                read_dict_of_lists["read_name"].append(read_name)
                read_dict_of_lists["chromosome"].append(read_chrom)
                read_dict_of_lists["read_start"].append(read_start)
                read_dict_of_lists["read_end"].append(read_end)
                read_dict_of_lists["strand"].append(ref_strand)
                read_dict_of_lists["motif"].append(motif)
                read_dict_of_lists["mod_vector"].append(
                    np.frombuffer(
                        gzip.compress(
                            mod_vector.tobytes(), compresslevel=compress_level
                        ),
                        dtype=np.uint8,
                    )
                )
                read_dict_of_lists["val_vector"].append(
                    np.frombuffer(
                        gzip.compress(
                            val_vector.tobytes(), compresslevel=compress_level
                        ),
                        dtype=np.uint8,
                    )
                )
                for dataset, entry in read_dict_of_lists.items():
                    h5[dataset][
                        old_size
                        + (read_counter // write_chunks) * write_chunks : old_size
                        + read_counter
                        + 1
                    ] = entry
                read_counter += 1
    return


def sanitize_path_args(*args) -> tuple:
    """
    Coerce all given arguments to Path objects, leaving Nones as Nones.
    """
    return tuple(Path(f) if f is not None else f for f in args)


def prep_outputs(
    output_directory: Path | None,
    output_name: str,
    input_file: Path,
    output_file_names: list[str],
) -> tuple[Path, list[Path]]:
    """
    As a side effect, if files exist that match the requested outputs, they are deleted.

    TODO: Is it kind of silly that this takes in input_file? Maybe should take in some generic default parameter, or this default should be set outside this method?
    Args:
        output_directory: Path pointing to an output directory.
            If left as None, outputs will be stored in a new folder within the input
            directory.
        output_name: a string that will be used to create an output folder
            containing the intermediate and final outputs, along with any logs.
        input_file: Path to input file; used to define default output directory
        output_file_names: list of names of desired output files

    Returns:
        * Path to top-level output directory
        * List of Paths to requested output files
    """
    if output_directory is None:
        output_directory = input_file.parent
        print(f"No output directory provided, using input directory {output_directory}")

    output_path = output_directory / output_name

    output_files = [output_path / file_name for file_name in output_file_names]

    # Ensure output path exists, and that any of the specified output files do not already exist (necessary for some outputs)
    output_path.mkdir(parents=True, exist_ok=True)
    for output_file in output_files:
        output_file.unlink(missing_ok=True)

    return output_path, output_files