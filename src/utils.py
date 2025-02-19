import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

MAPPING_DICT = {"A": 1, "C": 2, "G": 3, "T": 4}
INVERSE_MAPPING_DICT = {1: "A", 2: "C", 3: "G", 4: "T"}

def DNA2Signal(sequence: str) -> np.array:
    s = np.zeros(len(sequence))
    for i, nt in enumerate(sequence):
        s[i] = MAPPING_DICT[nt]
    return s

def Signal2DNA(signal):
    return "".join([INVERSE_MAPPING_DICT[num] for num in signal])

def display_sequences(sequences: Union[List[str], str]):

    if isinstance(sequences, str):
        sequences = [sequences]

    num_subplots = len(sequences)
    
    # Create subplots
    fig, axs = plt.subplots(num_subplots, 1)

    for i, ax in enumerate(axs):
        signal = DNA2Signal(sequences[i])
        ax.plot(signal)
        ax.set_title(f"Sequence {i}")
        ax.set_xlabel("Position")
        ax.set_ylabel("nt")

    plt.tight_layout()
    plt.show()

def viz(x, Tx, Wx):
    """Visualize colormap of a sequence x"""
    plt.imshow(np.abs(Wx), aspect='auto', cmap='turbo')
    plt.show()
    plt.imshow(np.abs(Tx), aspect='auto', vmin=0, vmax=.2, cmap='turbo')
    plt.show()


def get_resnet_for_fine_tuning(num_classes: int) -> resnet50:

    pretrained_resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Set all requires_grad to false
    for param in pretrained_resnet.parameters():
        param.requires_grad = False

    # Override the fc layer with the number of classes
    pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, num_classes)

    # Set last layer and fc requires_grad to True
    for param in pretrained_resnet.layer4.parameters():
        param.requires_grad = True
    
    for param in pretrained_resnet.fc.parameters():
        param.requires_grad = True
    
    # Return the new model
    return pretrained_resnet

def extract_sequences_FASTA(fastafile: str):
    with open(fastafile, 'r') as fopen:
        return [line.upper().strip() for line in fopen if not line.startswith('>')]
