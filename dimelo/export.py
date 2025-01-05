import os
from collections import deque
from pathlib import Path

import pyBigWig
import pysam
from tqdm.auto import tqdm

from . import utils

"""
This module contains code to export indexed and compressed parse output files to other formats that may be helpful for downstream analysis.
"""


def tail(n, iterable):
    """
    Return an iterator over the last n items.
    Copied from https://docs.python.org/3/library/itertools.html#itertools-recipes
    """

    # tail(3, 'ABCDEFG') → E F G
    return iter(deque(iterable, maxlen=n))


def pileup_to_bigwig(
    bedmethyl_file: str | Path,
    motif: str,
    bigwig_file: str | Path | None = None,
    strand: str = ".",
    chunk_size: int = 1000,
):
    """
    Extract a single motif from a pileup and write its mod fractions by position to a bigwig file.

    This function will take the entire contents of the pileup bedmethyl file and create a bigwig header with all of the same contigs, with
    contig lengths in the bigwig header set to the highest motif coordinate for each contig. If strand is specified as + or -, only that
    strand will be written to the output bigwig - this can allow for strand bias analysis in a genome browser. If strand is specified as .,
    as is the default, both strands will be included.

    The operation can be quite slow for large pileups. The current design is that if you want to create a bigwig for a subset of the genome,
    you can specify the regions at parsing time, rather than re-implementing the subset handling logic here.

    Args:
        bedmethyl_file: Path to the input bedmethyl file
        motif: type of modification to extract data for
        bigwig_file: Path to the output bigwig destination. If unspecified, a pileup.bw file will be created in the bedmethyl file's directory
        strand: the DNA strand to extra, + or - for forward or reverse and . for both
    """

    output_file_path = (
        bigwig_file
        if bigwig_file is not None
        else Path(bedmethyl_file).parent / "pileup.fractions.bigwig"
    )
    os.makedirs(output_file_path.parent, exist_ok=True)

    # Because we need to set up the bigwig header for we start writing data to it, we need to pre-index the length of each contig
    tabix = pysam.TabixFile(str(bedmethyl_file))
    contig_lengths_tuples = []
    lines_by_contig = {}

    parsed_motif = utils.ParsedMotif(motif)

    for contig in tqdm(
        tabix.contigs,
        desc=f"Step 1: Indexing contigs in {Path(bedmethyl_file).name} to set up bigwig header for {Path(output_file_path).name}",
    ):
        # count up the number of rows, for progress tracking, and pull out the last row so as to grab the length of the chromosome
        # note: the tqdm progress bar slows things down by about 33%, which was deemed better at the time of writing this than
        # 90 seconds without any status updates
        rows_count, last_row = list(
            tail(
                n=1,
                iterable=enumerate(
                    tqdm(
                        tabix.fetch(contig),
                        mininterval=1.0,
                        desc=f"Indexing {contig}.",
                        leave=False,
                    )
                ),
            )
        )[0]
        fields = last_row.split("\t")
        max_coord = int(fields[2])
        contig_lengths_tuples.append((contig, max_coord))
        lines_by_contig[contig] = rows_count

    with pyBigWig.open(str(output_file_path), "w") as bw:
        bw.addHeader(contig_lengths_tuples)
        for contig in tqdm(
            tabix.contigs,
            desc=f"Step 2: Writing {Path(bedmethyl_file).name} contents to {Path(output_file_path).name}",
        ):
            contig_list = []
            start_list = []
            end_list = []
            values_list = []
            for row in tqdm(
                tabix.fetch(contig),
                desc=f"Writing {contig}.",
                total=lines_by_contig[contig],
                leave=False,
            ):
                # TODO: This code is copied from load_processed.pileup_counts_from_bedmethyl and should probably be consolidated at some point
                tabix_fields = row.split("\t")
                pileup_basemod = tabix_fields[3]
                pileup_strand = tabix_fields[5]
                keep_basemod = False
                if (strand != ".") and (pileup_strand != strand):
                    # This entry is on the wrong strand - skip it
                    continue
                elif len(pileup_basemod.split(",")) == 3:
                    pileup_modname, pileup_motif, pileup_mod_coord = (
                        pileup_basemod.split(",")
                    )
                    if (
                        pileup_motif == parsed_motif.motif_seq
                        and int(pileup_mod_coord) == parsed_motif.modified_pos
                        and pileup_modname in parsed_motif.mod_codes
                    ):
                        keep_basemod = True
                elif len(pileup_basemod.split(",")) == 1:
                    if pileup_basemod in parsed_motif.mod_codes:
                        keep_basemod = True
                else:
                    raise ValueError(
                        f"Unexpected format in bedmethyl file: {row} contains {pileup_basemod} which cannot be parsed."
                    )
                # TODO: consolidate the above into a function; just do adding outside
                if keep_basemod:
                    pileup_info = tabix_fields[9].split(" ")
                    valid_base_counts = int(pileup_info[0])
                    modified_base_counts = int(pileup_info[2])
                    if valid_base_counts > 0:
                        genomic_coord = int(tabix_fields[1])
                        contig_list.append(contig)
                        start_list.append(genomic_coord)
                        end_list.append(genomic_coord + 1)
                        values_list.append(modified_base_counts / valid_base_counts)

                        if len(values_list) > chunk_size:
                            bw.addEntries(
                                contig_list,  # Contig names
                                start_list,  # Start positions
                                ends=end_list,  # End positions
                                values=values_list,  # Corresponding values
                            )
                            contig_list = []
                            start_list = []
                            end_list = []
                            values_list = []
            bw.addEntries(
                contig_list,  # Contig names
                start_list,  # Start positions
                ends=end_list,  # End positions
                values=values_list,  # Corresponding values
            )
