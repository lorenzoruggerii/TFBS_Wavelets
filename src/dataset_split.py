"""
"""

from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
from time import time
from glob import glob

import pandas as pd
import numpy as np

import pybedtools
import subprocess
import random
import sys
import os

# set seed for reproducibility
random.seed(1234)

# limit the analysis to consider only data mapped on canonical chromosomes
CHROMS = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
# train and test data directory names
TRAINDIR = "traindata"
TESTDIR = "testdata"


def parse_commandline(args: List[str]) -> Tuple[List[str], str, str]:
    if len(args) != 3:
        raise ValueError(f"Too many/few input arguments ({len(args)})")
    posdir, negdir, genome = args  # recover folders storing positive and negative data
    if not os.path.isdir(posdir):
        raise FileNotFoundError(f"Unable to locate positive data folder {posdir}")
    if not os.path.isdir(negdir):
        raise FileNotFoundError(f"Unable to locate negative data folder {negdir}")
    # retrieve positive and negative BED files
    posbeds, negbeds = glob(os.path.join(posdir, "*.bed")), glob(
        os.path.join(negdir, "*.bed")
    )
    if not posbeds:
        raise FileNotFoundError(
            f"Positive data folder {posdir} does not contain BED files"
        )
    if not negbeds:
        raise FileNotFoundError(
            f"Negative data folder {negdir} does not contain BED files"
        )
    assert len(negbeds) == 1  # assumes that only one negative dataset is provided
    # check genome reference file existence
    if not os.path.isfile(genome):
        raise FileNotFoundError(f"Unable to locate reference genome {genome}")
    return posbeds, negbeds[0], genome


def construct_dirtree(datadir: str) -> Tuple[str, str, str, str]:
    # create the directory tree within data folder
    assert os.path.isdir(datadir)
    traindir, testdir = os.path.join(datadir, TRAINDIR), os.path.join(datadir, TESTDIR)
    for d in [traindir, testdir]:  # train and test directories
        if not os.path.isdir(d):  # if not already present, create directory
            os.makedirs(d)
    # positive and negative train and test directories
    trainposdir, trainnegdir = os.path.join(traindir, "positive"), os.path.join(
        traindir, "negative"
    )
    testposdir, testnegdir = os.path.join(testdir, "positive"), os.path.join(
        testdir, "negative"
    )
    for d in [trainposdir, trainnegdir, testposdir, testnegdir]:
        if not os.path.isdir(d):  # if not already present, create directory
            os.makedirs(d)
    return trainposdir, trainnegdir, testposdir, testnegdir


def compute_seqname(chrom: str, start: int, stop: int) -> str:
    return f"{chrom}:{start}-{stop}"


def remove_duplicates(bedfile: str, fname: str, outdir: str) -> str:
    bedunique = os.path.join(outdir, f"{fname}_unique.bed")  # unique peaks fname
    bed = pd.read_csv(bedfile, sep="\t", header=None)
    seqnamescol = bed.shape[1]  # column containing seqnames
    # assign seqname to each peak and use the seqnames as keys for duplicate removal
    bed[seqnamescol] = bed.apply(
        lambda x: compute_seqname(x.iloc[0], x.iloc[1], x.iloc[2]), axis=1
    )
    # if duplicate peaks are detected, keep the one with higher enrichment score
    bed = bed.loc[bed.groupby(seqnamescol)[6].idxmax()]
    bed.to_csv(bedunique, sep="\t", header=None, index=False)
    return bedunique


def read_bed(bedfile: str) -> pybedtools.BedTool:
    # use pybedtools to read input and bed and perform effcient operations
    return pybedtools.BedTool(bedfile)


def subtract(a: pybedtools.BedTool, b: pybedtools.BedTool) -> pybedtools.BedTool:
    # remove features in a even if partially overlapping features in b
    c = a.subtract(b, A=True)
    return c


def filter_negative(positive: str, negative: str, fname: str, trainnegdir: str) -> str:
    # remive features in negative dataset even if partially overlapping with
    # features in positive dataset
    negative_filt_bed = subtract(read_bed(negative), read_bed(positive))
    negative_filt_fname = os.path.join(trainnegdir, f"{fname}_neg_no_overlap.bed")
    with open(negative_filt_fname, mode="w") as outfile:
        outfile.write(str(negative_filt_bed))
    return negative_filt_fname


def split_train_test(
    bedfile: str, testchrom: str, fname: str, traindir: str, testdir: str
) -> Tuple[str, str]:
    bed = pd.read_csv(bedfile, sep="\t", header=None)
    bed = bed[bed[0].isin(CHROMS)]  # remove data mapped in non canonical chroms
    # split dataset
    bed_train, bed_test = bed[bed[0] != testchrom], bed[bed[0] == testchrom]
    assert (bed_train.shape[0] + bed_test.shape[0]) == bed.shape[0]
    bed_train_fname = os.path.join(traindir, f"{fname}_train.bed")
    bed_train.to_csv(bed_train_fname, sep="\t", header=False, index=False)
    bed_test_fname = os.path.join(testdir, f"{fname}_test.bed")
    bed_test.to_csv(bed_test_fname, sep="\t", header=False, index=False)
    subprocess.call(f"rm {bedfile}", shell=True)  # remove old bed
    return bed_train_fname, bed_test_fname


def extract_sequences(bedfile: str, genome: str) -> str:
    bed = pybedtools.BedTool(bedfile)  # load bedtool object
    sequences = bed.sequence(fi=genome)  # extract sequences from reference
    fasta = f"{os.path.splitext(bed.fn)[0]}.fa"
    with open(fasta, mode="w") as outfile:
        outfile.write(open(sequences.seqfn).read())


def chrom_split(fname: str) -> Dict[str, List[str]]:
    chrom_dict = {c: [] for c in CHROMS}
    with open(fname, mode="r") as infile:
        for line in infile:
            chrom = line.strip().split()[0]
            chrom_dict[chrom].append(line)
    return chrom_dict


def compute_background_data(
    trainpos: str,
    trainneg: str,
    testpos: str,
    testneg: str,
    testchrom: str,
    fname: str,
    trainnegdir: str,
    testnegdir: str,
):
    # divide genomic features by chromosome
    trainpos_chrom, testpos_chrom = chrom_split(trainpos), chrom_split(testpos)
    trainneg_chrom, testneg_chrom = chrom_split(trainneg), chrom_split(testneg)
    bgtrain, bgtest = [], []
    for chrom in trainpos_chrom:  # select train features
        bgtrain.extend(random.sample(trainneg_chrom[chrom], len(trainpos_chrom[chrom])))
    # select test features
    test_th = len(testpos_chrom[testchrom])
    test_th = (
        test_th
        if len(testneg_chrom[testchrom]) > test_th
        else len(testneg_chrom[testchrom])
    )
    bgtest.extend(random.sample(testneg_chrom[testchrom], test_th))
    trainneg_fname = os.path.join(trainnegdir, f"{fname}_neg_train.bed")
    with open(trainneg_fname, mode="w") as outfile:
        outfile.write("".join(bgtrain))
    testneg_fname = os.path.join(testnegdir, f"{fname}_neg_test.bed")
    with open(testneg_fname, mode="w") as outfile:
        outfile.write("".join(bgtest))
    # remove old bed files
    for f in [trainneg, testneg]:
        subprocess.call(f"rm {f}", shell=True)
    return trainneg_fname, testneg_fname


def split_dataset(
    positive: str,
    negative: str,
    genome: str,
    trainposdir: str,
    trainnegdir: str,
    testposdir: str,
    testnegdir: str,
):
    # retrieve experiment name
    chip_fname = os.path.splitext(os.path.basename(positive))[0]
    # remove potential duplicate peaks from positive sequences
    positive = remove_duplicates(positive, chip_fname, trainposdir)
    # remove features from negative overlapping with features on positive
    negative = filter_negative(positive, negative, chip_fname, trainnegdir)
    # split positive dataset in train and test
    trainpos, testpos = split_train_test(
        positive, "chr2", chip_fname, trainposdir, testposdir
    )
    extract_sequences(trainpos, genome)
    extract_sequences(testpos, genome)
    # split negative dataset in train and test
    trainneg, testneg = split_train_test(
        negative, "chr2", chip_fname, trainnegdir, testnegdir
    )
    trainneg, testneg = compute_background_data(
        trainpos,
        trainneg,
        testpos,
        testneg,
        "chr2",
        chip_fname,
        trainnegdir,
        testnegdir,
    )
    extract_sequences(trainneg, genome)
    extract_sequences(testneg, genome)
    # remove old bed files
    for f in [trainpos, testpos, trainneg, testneg]:
        subprocess.call(f"rm {f}", shell=True)


def center_bedfile(bedfile: str):
    """Centers bedfile in the peak summit position"""
    bed = pd.read_csv(bedfile, sep="\t", header=None)
    pk = bed[9] # peak summit
    seqs_len = bed[2] - bed[1] # start - stop
    max_dists = np.maximum(pk, seqs_len - pk)
    new_start = bed[1] + pk - max_dists
    new_end = bed[1] + pk + max_dists
    bed[1] = new_start
    bed[2] = new_end
    bed.to_csv(bedfile, sep = "\t", header=None, index=False)

def main():
    # parse command line arguments -> expected input: folder containing positive
    # sequence dataset (BED format) and folder containing negative sequence
    # dataset (BED format)
    posbeds, negbed, genome = parse_commandline(sys.argv[1:])
    # construct train and test data directory tree
    trainposdir, trainnegdir, testposdir, testnegdir = construct_dirtree("../dataset/data")
    # split each dataset on train and test
    sys.stdout.write("Train and test datasets construction\n")
    start = time()
    for posbed in tqdm(posbeds):
        center_bedfile(posbed)
        split_dataset(
            posbed, negbed, genome, trainposdir, trainnegdir, testposdir, testnegdir
        )
    sys.stdout.write(
        f"Train and test datasets construction completed in {(time() - start):.2f}s\n\n"
    )


if __name__ == "__main__":
    main()
