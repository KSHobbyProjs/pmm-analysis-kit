#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src import io

def _parse_ks(kstr):
    s = kstr.strip()
    
    # assume range given as kmin:kmax
    if ":" in s:
        kmin, kmax = s.split(":", 1)
        return list(np.arange(int(kmin), int(kmax) + 1))
    
    # else assume comma-separated list
    if "," in s:
        return [int(k.strip()) for k in s.split(",")]

    # else assume single int
    return [int(kstr)]

def _parse_linestyles(linestyles, num_files):
    linestyles = linestyles.strip().split(",")
    # if linestyles length is too short for files length, extend:
    if num_files  > len(linestyles):
        dif = num_files - len(linestyles)
        linestyles += [linestyles[-1]] * dif
    linestyles = [ls.strip() for ls in linestyles] # strip each element for security
    return linestyles

def main():
    parser = argparse.ArgumentParser(description="Plot energies read from a file.")
    parser.add_argument("-f", "--files", nargs="+", required=True, help="Paths to input files.")
    parser.add_argument("-k", "--knum", type=str, default='0', help="Number of energies to plot per parameter.")
    parser.add_argument("-l", "--linestyle", type=str, default="-,--", help="Linestyle for each plot.")
    parser.add_argument("-q", "--quiet", action="store_false", help="Omit legend when plotting.")
    parser.add_argument("-o", "--out", type=str, default=None, help="Output png file name.")
    parser.add_argument("-n", "--no-show", action="store_true", help="Don't show figure after plotting.")
    args = parser.parse_args()

    # parse linestyles
    linestyles = _parse_linestyles(args.linestyle, len(args.files))
    # parse knum range
    knum = _parse_ks(args.knum)

    fig, ax = plt.subplots()
    for i, file in enumerate(args.files):
        if file.endswith(".h5"):
            Ls, energies, _, _ = io.load_energies_from_h5(file)
        else:
            Ls, energies, _ = io.load_energies_from_dat(file)
        kidx = [k for k in knum if k < energies.shape[1]] # filter out out-of-bounds indices silently
        ax.plot(Ls, energies[:, kidx], f'{linestyles[i]}', label=f"{file}")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Energies")
    ax.set_title("Energies vs Parameters")
    if args.quiet:
        plt.legend()
    if args.out:
        plt.savefig(args.out)
    if not args.no_show:
        plt.show()
    plt.close()

if __name__=="__main__":
    main()
