#!/usr/bin/env python3
import sys
import argparse
import time
import logging
logger = logging.getLogger(__name__)

from src import parse, io

def _setup_logging(verbose=0):
    if verbose == 0: 
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
            level = level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        )

def _parse_args(): 
    parser = argparse.ArgumentParser(description="Compute eigenpairs at a set of parameters L for a given model.")
    parser.add_argument("-m", "--model", type=str, default="gaussian.Gaussian1d:N=128,V0=-4.0,R=2.0", help="Model in 'module.Class:kw1=val,kw2=val' format.")
    parser.add_argument("-L", "--parameters", type=str, default="5.0,20.0:20", help="Parameter range in 'start,end:len', '1.0,2.0,3.0', or 'start,end:len,exp' format.")
    parser.add_argument("-k", "--knum", type=int, default=1, help="Number of eigenvalues to compute at each parameter value.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output filename (optional).")
    parser.add_argument("--vectors", action="store_true", help="Output eigenvectors as well as eigenvalues.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    parser.add_argument("-q", "--quiet", action="store_true", help="Turn of print output.")
    args = parser.parse_args()
    return args

def _parse_parameters_and_model(parameter_str, model_str):
    Ls = parse.parse_parameter_values(parameter_str)
    model_instance = parse.parse_model_instance(model_str)
    logger.debug(f"Parsed {model_str} -> " +
                 f"{type(model_instance).__name__}(" + ','.join(f"{k[1:]}={v} ({type(v).__name__})" for k, v in vars(model_instance).items()) + ").")
    logger.debug(f"Parsed {parameter_str} -> " + 
                 f"{type(Ls).__name__}, min={min(Ls)}, max={max(Ls)}, len={len(Ls)}.")
    return Ls, model_instance

def _write_results(args, Ls, eigenvalues, eigenvectors):
    # create metadata
    metadata = {"timestamp"      : time.strftime("%A, %b %d, %Y %H:%M:%S"),
                "model"      : args.model,
                "parameters" : args.parameters,
                "command"        : ' '.join(sys.argv)
                }
    # if file ends in h5, use h5py format, otherwise treat everything like .dat
    if args.output.endswith(".h5"):
        io.write_energies_to_h5(args.output, Ls, eigenvalues, eigenvectors if args.vectors else None, metadata)
        logging.debug("Finished writing HDF5 file.")
    else:             
        if args.vectors:
            logger.warning("You selected --vectors but chose an output file that does not support storing full eigenvectors. "
                  "Use a .h5 file if you want full eigenvector output.")
        io.write_energies_to_dat(args.output, Ls, eigenvalues, metadata)
        logging.debug("Finished writing text output file.")

def _print_results(args, Ls, eigenvalues, eigenvectors):
    trnc = 9 # num of eigenvalues to print before skipping rest
    vectrnc = 4
    for i, L in enumerate(Ls):
        print(f"Spectrum at L = {L:.3f}")
        print(f"\t[ " + ", ".join(f"{eigval:.6f}" for eigval in eigenvalues[i][:trnc]), end="")
        
        if args.knum > trnc: 
            print(f", ..., {eigenvalues[i][-1]:.6f} ] ({len(eigenvalues[i])} energies total)")
        else: 
            print(" ]")
        
        if args.vectors:
            for k, vec in enumerate(eigenvectors[i][:vectrnc]):
                # print only the first 4 elements of each eigenvector
                # (prints all elements if 4 > len(vec))
                formatted_eigenvecs = [f"{v.real:.4f} + {v.imag:.4f}j" for v in vec[:4]]
                print(f"\tEigenvector {k}: {formatted_eigenvecs} ... ({len(vec)} entries total)")
            
            if args.knum > vectrnc:
                print(f"\t... ({len(eigenvectors[i])} eigenvectors total).")

def main():
    # parse args, setup logging, and start timer
    args = _parse_args()
    _setup_logging(args.verbose)
    start = time.time()          
    print(f"Model = {args.model.strip()}\tLs = {args.parameters.strip()}\tk = {str(args.knum).strip()}")

    # parse parameters and model instance
    logger.info(f"Parsing parameters = {args.parameters} and model = {args.model}.")
    Ls, model_instance = _parse_parameters_and_model(args.parameters, args.model)

    # compute eigenpairs
    logger.info("Computing eigenvalues.")
    eigenvalues, eigenvectors = model_instance.get_eigenvectors(Ls, args.knum) 

    # write eigenpair data to file if requested
    if args.output:
        logger.info(f"Writing results to {args.output}.")
        _write_results(args, Ls, eigenvalues, eigenvectors)

    # print results
    if not args.quiet:
        _print_results(args, Ls, eigenvalues, eigenvectors)
    end = time.time()
    print(f"Done.\nElapsed time: {end - start:.3f} seconds.")

if __name__=="__main__":
    main()
