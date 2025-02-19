import numpy as np
import matplotlib.pyplot as plt
from typing import List
from ssqueezepy import ssq_cwt
import sys
from utils import DNA2Signal, extract_sequences_FASTA
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_sequence(args):
    """Processes a single sequence: converts to signal, applies transform, and saves the colormap."""
    seq, i, outdir, synchrosqueeze, prefix = args
    
    # Transform sequence to signal
    signal = DNA2Signal(seq)

    # Synchro squeezing wavelet transform
    Twxo, Wxo, *_ = ssq_cwt(signal)
    transform = Twxo if synchrosqueeze else Wxo

    # Create the colormap and save it into outdir
    plt.figure(figsize=(12, 8))
    plt.imshow(np.abs(transform), aspect='auto', cmap='turbo')
    plt.axis('off')
    plt.savefig(os.path.join(outdir, f"{prefix}_cmap_{i}.png"), bbox_inches='tight')
    plt.close()

def main():
    fastafile, outdir, synchrosqueeze, prefix, threshold, num_workers = sys.argv[1:]
    os.makedirs(outdir, exist_ok=True)
    
    synchrosqueeze = int(synchrosqueeze)  # 1 if using synchrosqueezing, 0 otherwise
    threshold = int(threshold)
    num_workers = int(num_workers)

    sequences = extract_sequences_FASTA(fastafile)

    if threshold < len(sequences):
        sequences = sequences[:threshold]

    # Prepare arguments for multiprocessing
    args_list = [(seq, i, outdir, synchrosqueeze, prefix) for i, seq in enumerate(sequences)]

    # Use multiprocessing Pool
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_sequence, args_list), total=len(sequences), colour="GREEN"))

if __name__ == '__main__':
    main()
