#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src import io


def main():
    parser = argparse.ArgumentParser(description="Plot energies read from a file.")
    parser.add_argument("-f", "--files", nargs="+", required=True, help="Paths to input files.")
    parser.add_argument("-k", "--knum", type=int, default=1, help="Number of energies to plot per parameter.")
    parser.add_argument("-l", "--linestyle", type=str, default="-,--", help="Linestyle for each plot.")
    args = parser.parse_args()

    # parse linestyles
    linestyles = args.linestyle.split(",")
    # if linestyles length is too short for files length, extend:
    if len(args.files) > len(linestyles):
        dif = len(args.files) - len(linestyles)
        linestyles += ['-'] * dif

    fig, ax = plt.subplots()
    for i, file in enumerate(args.files):
        if file.endswith(".h5"):
            Ls, energies, _, _ = io.load_energies_from_h5(file)
        else:
            Ls, energies, _ = io.load_energies_from_dat(file)
        ax.plot(Ls, energies[:, :args.knum], f'{linestyles[i]}', label=f"{file}")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Energies")
    ax.set_title("Energies vs Parameters")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
