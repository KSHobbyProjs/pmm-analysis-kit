#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src import io


def main():
    parser = argparse.ArgumentParser(description="Plot energies read from a file.")
    parser.add_argument("input_file", type=str, help="Path to input file.")
    args = parser.parse_args()
    
    data = np.loadtxt(args.input_file)
    epochs = data[:, 0]
    losses = data[:, 1]

    fig, ax = plt.subplots()
    ax.plot(epochs, np.log10(losses))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Losses")
    ax.set_title("Losses vs Epochs")
    plt.show()

if __name__=="__main__":
    main()
