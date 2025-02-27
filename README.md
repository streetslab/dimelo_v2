# dimelo 0.2.0 beta
## Introduction

This branch is a beta version of a rebuilt-from-scratch dimelo package that is still in development. As of February 2024, exact functionality and function interfaces / parameters are not yet finalized. Final functionality and design will be in part driven by user feedback.

We support the same core pileup and single read extraction operations as the original `dimelo` package, but have focused on a number of objectives in building this new version, many of which were impossible or cumbersome to implement with the previous design:

<ol>
    <li>Support multicolor data / any base modification context (GpC, CpC, etc)</li>
    <li>Vector extraction for all data types</li>
    <li>Enhanced speed and reliability, enabling e.g. whole genome processing</li>
    <li>Maintainability -> using a small number of standard dependencies, outsourcing as much as possible to well-maintained third-party packages (e.g. modkit, pysam, h5py, and a few others)</li>
    <li>Modularity in both architecture and operation</li>
    <li>Ease of use, especially for multiplatform installation</li>
    <li>More powerful plotting e.g. bam files from different basecallers, single read sorting, rapid iteration</li>
</ol>

This README document contains installation instructions and documentation for various use cases. There is a [tutorial](#basic-use) jupyter notebook that will take you through the core functionality of the package step-by-step. For Google Colab, the notebook already contains the necessary code to set up and run `dimelo`, whereas for local operation you will first need to follow the [local install instructions](#local-install-via-conda). Be sure to check that your system meets our [specifications](#system-requirements).

## Contents
[1.0 Install Instructions](#Install-instructions)

-[1.1 Local Install via Conda](#Local-Install-via-Conda)
    
-[1.2 Google Colab Installation](#Google-Colab-Installation)

-[1.3 Alternative Installations](#Alternative-Installations)

-[1.4 Developer Installation](#Developer-Installation)

[2.0 Basic Use](#Basic-Use)

-[2.1 Parameters and what they mean](#Parameters-and-what-they-mean)
    
-[2.2 Parsing and processing](#Parsing-and-processing)

-[2.3 Plotting](#Plotting)

-[2.4 Load values from processed files](#load-values-from-processed-files)

[3.0 Known Issues](#known-issues)

-[3.1 No progress bars](#no-progress-bars)

# Install Instructions
## Local Install via Conda

### System Requirements

**System Specs:** You will need at least 10GB of disk space and 10GB of RAM for the tutorial (largely due to the reference genome). More disk space may be required if processing large datasets. Additionally, if you want to run on many cores, you should have at least 4GB of RAM per core. If you have less per-core memory than this, consider specifying a subset of cores when calling parsing methods. See the [parameters](#parameters-and-what-they-mean) section for more information. `cores=1` will use the least memory and thus is the least likely to be terminated by your OS.

**Platforms:** Mac and Linux operating systems, ARM (e.g. M1/M2 mac) and x86 (e.g. Intel mac) architectures. The package has been tested on HPC clusters, but there may be additional complexities depending on how these systems are set up.

*For Windows,  we recommend using [Google Colab](https://colab.research.google.com/). We have not tested on [Windows Linux Subsystem](https://learn.microsoft.com/en-us/windows/wsl/install) but in principle that should work too. Windows support is possible in future, but blocked by [conda availability for modkit executables](https://anaconda.org/nanoporetech/modkit) and the [current implementation](dimelo/run_modkit.py) of live error/progress tracking during modkit execution, which relies on a unix-only library as of Python 3.11. The urgency of a Windows implementation will depend on user need, so please let us know if this is important for you.*

**Conda and Python:** The default installation requires conda, or alternatives like mamba. See [here](https://www.anaconda.com/download) for conda installation. The installation instructions below will install Python 3.11 for you within a conda virtual environment, but depending on your system configuration you may need to ensure that you are not also loading a different version of Python on your path. If you encounter unexpected errors when importing `dimelo`, e.g. complaining about syntax, consider checking your Python version.

### Load source code from the modkit_parsing_beta branch

Open your terminal or command line and navigate to wherever you want to keep the `dimelo_v2` source code (e.g. your Documents folder, `cd Documents`) and clone the repo

```
git clone https://github.com/streetslab/dimelo_v2
```

### Set up virtual environment

Navigate into the dimelo_v2 directory

```
cd dimelo_v2
```

Create a conda environment using environment.yml. This will make a new conda environment with the name `dimelo_modkit_parsing`. 

```
conda env create -f environment.yml
```

*If you want to handle environment creation yourself, see [the alternative installation instructions](#alternative-installations).*

### Install pip dependencies and core dimelo package

Activate your conda environment, which should now contain python 3.11 and a modkit executable on the path and executable on your system.

```
conda activate dimelo_modkit_parsing
```

Ensure that you are still in the top-level dimelo_v2 directory. Install the dimelo package and its dependencies from source.

```
pip install .
```

## Google Colab Installation

Run the following code in the first cell of your notebook to grab `modkit v0.2.4` from conda and install the `dimelo modkit_parsing_beta` branch. This will have to be run whenever you make a new Colab instance, unless you have a better way of managing this, in which case please reach out. The tutorial notebook runs equivalent code blocks to set up your environment, so if you are trying to run the tutorial you can skip to [Basic Use](#basic-use).

```
from google.colab import drive
drive.mount('/content/drive')
!pip install -q condacolab
import condacolab
condacolab.install()
!conda install nanoporetech::modkit==0.2.4
!git clone https://github.com/streetslab/dimelo_v2
!cd dimelo_v2 && pip install ipywidgets==7.7.1 .
import dimelo
```

## Alternative Installations

Alternatively, you can install modkit into any conda environment you like. If you want to, you can install modkit some other way, and then add it to the path of your notebook or script. *NOTE: if you are creating the environment yourself, be sure to use python 3.10 or greater. Some dimelo package features require relatively new python releases.*

```
conda install nanoporetech::modkit==0.2.4
```
OR
```
# install modkit some other way
# add to path in python before importing dimelo
import sys
sys.path.append('path_to_modkit_executable_directory')
```

## Developer Installation
If you are planning on developing for the `dimelo` package, change the `pip install` command to install the package in "editable" mode, so that your code changes are reflected in your environment:
```
pip install . -e
```

Additionally, be aware that this package uses [ruff](https://docs.astral.sh/ruff/) to enforce code standards. To make it easy to check that your changes meet standards, we provide [pre-commit](https://pre-commit.com/) hooks that run the checks automatically when you commit.

After installing [pre-commit](https://pre-commit.com/) on your system, run the following from the top level of the repository to set up the hooks:
```
pre-commit install
```

If you need to manually trigger a formatting check, the following command will forcibly run all checks on the entire repository:
```
pre-commit run --all-files
```

To run functionality tests on your machine, ensure that `pytest` is installed in your conda environment:
```
conda install pytest
```

The tests can be run from the top level of the repository using the following command:
```
pytest
```

# Basic Use

See the [tutorial](tutorial.ipynb) as a starting point.

For local operation on Mac or Linux, you will already have cloned the repo to disk in the installation step. Activate your conda environment, make sure you have jupyter installed, and then launch a jupyter notebook server and navigate to `tutorial.ipynb`. You can also use other tools to open the jupyter notebook or you can simply reference it as an example.

```
conda activate dimelo_modkit_parsing
jupyter notebook
```

If you want to run the tutorial on Google Colab, you can download [tutorial.ipynb](tutorial.ipynb), upload it to [Google Colab](https://colab.research.google.com/), and follow the instructions in the cells.

## Parsing and processing

The general workflow of this package is as follows:
```
Parsing: aligned modbam file (latest .bam spec) --> processed file
Loading: processed file --> python objects
Plotting: python objects --> visualizations
```

Both pileup and extract are typically run with a .bed file of regions, which can then be also passed to the plotting functions. All regions are processed into a file called `regions.processed.bed` which follows the format required by `modkit`:
```
chr14	44123158	44123308	+  .  .
```

`parse_bam.pileup` creates a bedmethyl genome-position-wise pileup for profiles and enrichment plotting between regions/modifications or for pulling out a genomic track at one or more regions.

```
def pileup(
    input_file: str | Path,
    output_name: str,
    ref_genome: str | Path,
    output_directory: str | Path = None,
    regions: str | Path | list[str | Path] = None,
    motifs: list = ['A,0','CG,0'],
    thresh: float = None,
    window_size: int = None,
    cores: int = None,
    log: bool = False,
    cleanup: bool = True,
    quiet: bool = False,
    override_checks: bool = False,) -> Path, Path:
```

`parse_bam.extract` creates an hdf5 file with datasets for different aspects of single read data, which can then be passed to plot single reads.
```
def extract(
    input_file: str | Path,
    output_name: str,
    ref_genome: str | Path,
    output_directory: str | Path = None,
    regions: str | Path | list[str | Path] = None,
    motifs: list = ['A,0','CG,0','GCH,1'],
    thresh: float = None,
    window_size: int = None,
    cores: int = None,
    log: bool = False,
    cleanup: bool = True,
    quiet: bool = False,
    override_checks: bool = False,) -> Path, Path:
```

For human-readable pileups (bedmethyl files, .bed) and extracted reads (.txt tab-separated values), run with `cleanup=False`. `cleanup=True` will clear these outputs because they can take up a lot of space.

### Parsing outputs
You should expect to see some text outputs and a series of progress bars. Progress bars tell you an estimated time remaining (typically an overestimate by 2-3x at the beginning of contig/chromosome). If you do not see progress bars, go to the [known issues: no progress bars](#no-progress-bars) section for possible fixes. 

There should not be such issues for command line operation. See below an example of command line progress outputs: you should expect relatively fast pre-processing, 10-90 seconds, and then contig processing times depending heavily on the size of your `.bam` file and the extent of your `regions`.

```
(dimelo_modkit_parsing) oberondixon-luinenburg@Oberons-MacBook-Pro package_test_notebooks % python dimelo_cmd.py
modkit found with expected version 0.2.4
No output directory provided, using input directory /Users/oberondixon-luinenburg/Documents/Ioannidis-Streets/dimelo_test_data/20230702_jm_lmnb1_acessibility_redux
No specified number of cores requested. 8 available on machine, allocating all.
Modification threshold of 0.9 will be treated as coming from range 0-1.
████████████████████| Preprocessing complete for motifs ['A,0'] in chm13.draft_v1.1.fasta:  100% | 00:30
███████████████████| All regions complete in mod_mappings.01.retagged.ma.sorted.bam:  100% | 02:23<00:00
████████████████| processed 218324 reads, 13323144 rows, skipped ~184465 reads, failed ~0 reads:  100%
```

You should see no outputs at all if `quiet=True`.

## Plotting

`plot_enrichment_profile` module for pileup line plot profiles across one or more region
```
def plot_enrichment_profile(mod_file_names: list[str | Path],
                            regions_list: list[str | Path | list[str | Path]],
                            motifs: list[str],
                            sample_names: list[str],
                            window_size: int,
                            smooth_window: int | None = None,
                            **kwargs) -> Axes:
def by_modification(mod_file_name: str | Path,
                    regions: str | Path,
                    motifs: list[str],
                    *args,
                    **kwargs) -> Axes:
def by_regions(mod_file_name: str | Path,
               regions_list: list[str | Path | list[str | Path]],
               motif: str,
               sample_names: list[str] = None,
               *args,
               **kwargs) -> Axes:
def by_dataset(mod_file_names: list[str | Path],
               regions: str | Path | list[str | Path],
               motif: str,
               sample_names: list[str] = None,
               *args,
               **kwargs) -> Axes:
```
`plot_enrichment` module for enrichment (e.g. mA/A) bar plot comparisons
```

def plot_enrichment(mod_file_names: list[str | Path],
                    regions_list: list[str | Path | list[str | Path]],
                    motifs: list[str],
                    sample_names: list[str],
                    **kwargs) -> Axes:
def by_modification(mod_file_name: str | Path,
                    regions: str | Path | list[str | Path],
                    motifs: list[str],
                    *args,
                    **kwargs) -> Axes:
def by_regions(mod_file_name: str | Path,
               regions_list: list[str | Path | list[str | Path]],
               motif: str,
               sample_names: list[str] = None,
               *args,
               **kwargs) -> Axes:
def by_dataset(mod_file_names: list[str | Path],
               regions: str | Path | list[str | Path],
               motif: str,
               sample_names: list[str] = None,
               *args,
               **kwargs) -> Axes:
```
`plot_reads` module for single read plots
```
def plot_reads(mod_file_name: str | Path,
               regions: str | Path | list[str | Path],
               motifs: list[str],
               window_size: int = None,
               sort_by: str | list[str] = 'shuffle',
               thresh: float = None,
               relative: bool = True,
               **kwargs
              ) -> Axes:
```
## Load values from processed files

`load_processed.pileup_counts_from_bedmethyl` for valid/modified counts from a specified region or set of regions
```
def pileup_counts_from_bedmethyl(bedmethyl_file: Path,
                          motif: str,
                          regions: str | Path | list[str | Path] = None,
                          ) -> tuple[int, int]:
```
`load_processed.pileup_vectors_from_bedmethyl` for valid over modified fraction from a specified region or set of regions
```
def pileup_vectors_from_bedmethyl(bedmethyl_file: str | Path,
                          motif: str,
                          regions: str | Path | list[str | Path],
                          window_size: int = None) -> (np.ndarray,np.ndarray):
```
`load_processed.read_vectors_from_hdf5` for read-by-basemod lists of valid and modified positions
```
def read_vectors_from_hdf5(
        file: str | Path,
        motifs: list[str],
        regions: str | Path | list[str | Path] = None,
        window_size: int = None,
        sort_by: str | list[str] = ['chromosome','region_start','read_start'],
        calculate_mod_fractions: bool = True,
) -> (list[tuple],list[str],dict):
```

## Parameters and what they mean

Many of the parsing, loading, and plotting functions share parameters. The common ones and their meanings/defaults are listed below:

`input_file` for parsing functions is a mandatory parameter providing the path to an aligned .bam file with modification calls, a .bam.bai index, and tags following the latest .bam specifications. parse_bam will check whether this .bam file meets the specifications and tell you what to do if it doesn't.

`output_name` for parsing functions is a mandatory string parameter that will be given to the new folder containing the processed outputs.

`ref_genome` for parsing functions is a mandatory parameter providing the path to the reference genome .fasta file to which the `input_file` .bam is aligned.

`output_directory` for parsing functions is an optional parameter specifying a parent directory for your outputs. By default, this will simply be the directory in which you `input_file` resides.

`mod_file_name` and `mod_file_names` for plotting functions are mandatory parameters providing a path to processed/parsed files that are ready for plotting. These paths are returns by the parsing functions but can also be provided manually by the user as a string or Path object. If providing manually, the path should point to a .bed.gz file with an accompanying .bed.gz.tbi index for profile and enrichment plots and to an .h5 file for read plots. `mod_file_name` points to a single file whereas `mod_files_names` is a list of files.

`regions` and `regions_list` are used for specifying subsets of the genome to parse, load, or plot. A `region` is defined as a range of genomic coordinates, and `regions` can refer to any number of `region` specifications. Thus for the `regions` parameter one can pass a single region specified as a string, `chrX:XXX-XXX`, many regions defined in a .bed tab-separated-value file with each line containing at miniimum chromosome, start, and end coordinates (plus optionally a strand, + or -), or a list of strings specifiers or bed files. The entire list will be rolled into a single `regions` set to be passed down for subsequent processing. In the case of regions-wise comparisons in plotting functions, `regions_list` is a *list of regions objects*, where each element of the list is a string, Path, or list of strings or Paths.

`motif` and `motifs` are used to specify what base modifications you are interested in and what their sequence context is for parse, load, and plot functions. A single `motif` is a string containing several canonical bases (using the [IUPAC nucleic acid notation](https://en.wikipedia.org/wiki/Nucleic_acid_notation), e.g. **H** refers to "not a G"), followed by a comma, and then an integer specifying which coordinate in the string is your modified base. For example, 6mA is denoted "A,0" and CpG is denoted "CG,0" whereas GpC *excluding CpGs* is denoted "GCH,1". `motifs` is a list of such strings for functions that can work on multiple base modifications at once.

`thresh` for parsing and some loading/plotting functions refers to a base modification probability threshold, used to transform the the output of most basecalling pipelines into a binary call for any given read position. For parsing pileup calls, this defaults to `None` which allows `modkit` to pick its own threshold based on the data. For other calls, the parameter is mandatory. The normal use is specifying between 0 and 1, but 1-255 is also supported to make the inputs more backwards compatible with old `dimelo` package versions and with examination of the raw .bam file contents. A value between and 255 will simply be converted into a 0-1 probability before being handed down to subsequent processing.

`window_size` for parsing and most loading and plotting functions is a *modification to your regions* that will redefine them to be all the same size, i.e. 2 x window_size, centered around the centers of your original regions. This is important for the parsing and plotting applications that show many genomic regions at once, but should be left blank if you don't want your regions modified. The default is `None` for functions where the parameter is optional. 

`cores`, `log`, `cleanup`, `quiet`, and `override_checks` can be ignored for most parsing applications. `cores` allows you to specify that `modkit` uses only a fraction of all the compute resources of your machine, rather than all; `log` will save the modkit logs for troubleshooting, and `cleanup` will keep the (often large) human-readable outputs that are inefficient for plotting and vector extraction but may be helpful for other use cases. `quiet` suppressed progress bars and other outputs and `override_checks` lets you run modkit even if the bam format checking and reference alignment checking are anomalous.

`relative` is a boolean input that specifies whether loading and plotting operations adjust coordinates to be relative to some center point or simple plot in absolute genomic coordinates.

`sort_by` for plot_reads will sort reads by any of `region_start`, `region_end`, `read_name`, `read_start`, `read_end`, `chromosome`, `strand`, and `MOTIF_mod_fraction` for any extracted motif. New sorting options are planned in the future. The default is `shuffle`, which will put the reads in a random order. `sort_by` can be passed as one string or as a list of strings. If a list is passed, the reads will be sorted hierarchically i.e. first by the first list element, then the second, and so on. The exception is that if any of the list elements are `shuffle`, the reads will *first* be shuffled and then sorted by the rest of the elements in order of priority.

`**kwargs` for all plotting functions get passed down to the underlying matplotlib / seaborn plotting functions. See matplotlib and seaborn documentation for more details.

# Known Issues
## No progress bars
The most common culprit for progress bar issues in notebooks (Jupyter or Colab) is an incompatibility between your notebooks interfaces and your `ipywidgets` version. The latest jupyter notebooks or jupyter lab install and the latest ipywidgets should work together, but on Google Colab, VS Code, Open On Demand, and other jupyter interfaces this may not be the case. [setup.py](setup.py) contains details on which versions you can try downgrading to for different platforms. The following code run in your activated conda environment will downgrade `ipywidgets` to your specified version. **Our Colab instructions in the [Colab Installation](#google-colab-installation) section and the [tutorial](tutorial.ipynb) already handle this for you.**

```
pip install ipywidgets==X.XX.X
```


