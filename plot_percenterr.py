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

def main():
    parser = argparse.ArgumentParser(description="Plot energies read from a file.")
    parser.add_argument("-e", "--exact", type=str, required=True, help="Path to file storing exact data.")
    parser.add_argument("-a", "--approximate", type=str, required=True, help="Path to file storing approximate data.")
    parser.add_argument("-k", "--knum", type=str, default='0', help="Number of energies to plot per parameter.")
    parser.add_argument("-q", "--quiet", action="store_false", help="Omit legend when plotting.")
    args = parser.parse_args()

    # parse knum range
    knum = _parse_ks(args.knum)
   
    # load files
    # load exact data
    if args.exact.endswith(".h5"):
        Ls_exact, energies_exact, _, _ = io.load_energies_from_h5(args.exact)
    else:
        Ls_exact, energies_exact, _ = io.load_energies_from_dat(args.exact)

    # load approximate data
    if args.approximate.endswith(".h5"):
        Ls_approx, energies_approx, _, _ = io.load_energies_from_h5(args.approximate)
    else:
        Ls_approx, energies_approx, _ = io.load_energies_from_dat(args.approximate)
  
    if not np.allclose(Ls_approx, Ls_exact):
        print("[WARNING]: Detected that the exact data and approx data might come from different parameter ranges.")
    Ls = Ls_exact

    # only include the ranges the user wants
    energies_exact = energies_exact[:, knum]
    energies_approx = energies_approx[:, knum]
    # calc percent difference
    percent_err = (np.abs(energies_exact - energies_approx) / np.abs(energies_exact)) * 100

    # plot
    fig, ax = plt.subplots()
    for i, k in enumerate(knum):
        ax.plot(Ls, percent_err[:, i], label=f'k={k}')
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Percent Error")
    ax.set_title("Percent Error vs Parameters")
    if args.quiet:
        plt.legend()
    plt.show()

if __name__=="__main__":
    main()
