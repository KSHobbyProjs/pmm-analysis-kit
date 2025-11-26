#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src import io


def main():
    parser = argparse.ArgumentParser(description="Plot energies read from a file.")
    parser.add_argument("input_file", type=str, help="Path to input file.")
    parser.add_argument("-o", "--out", type=str, default=None, help="Output png file name.")
    parser.add_argument("-n", "--no-show", action="store_true", help="Don't show figure after plotting.")
    args = parser.parse_args()
    
    data = np.loadtxt(args.input_file)
    epochs = data[:, 0]
    losses = data[:, 1]

    fig, ax = plt.subplots()
    ax.plot(epochs, np.log10(losses))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Losses")
    ax.set_title("Losses vs Epochs")
    if args.out:
        plt.savefig(args.out)
    if not args.no_show:
        plt.show()
    plt.close()

if __name__=="__main__":
    main()
