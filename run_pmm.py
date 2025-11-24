#!/usr/bin/env python3

import sys
import argparse
import time
import logging
logger = logging.getLogger(__name__)

from src import parse, io, utils

def _setup_logging(verbose=0):
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
            level = level,
            format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt = "%H:%M:%S"
        )
    # Silence JAX logs
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

def _parse_args():
    parser = argparse.ArgumentParser(description="Run a Parametric Matrix Model using energy data loaded from a file.")
    parser.add_argument("input_file", type=str, help="Path to the input file.")
    parser.add_argument("-p", "--pmm-name", type=str, default="PMM", help="Name of the PMM class to use.")
    parser.add_argument("-c", "--config", type=str, default="", help="Comma-separated key=val pairs to override default PMM parameters. E.g., eta=1.0e-2,beta1=0.9.")
    parser.add_argument("--config-file", type=str, default=None, help="Path to a file to load PMM parameters. key=val pairs passed through -c overwrite config files.")
    parser.add_argument("-o", "--save-energies", type=str, default=None, help="Path to file for energy data output.")
    parser.add_argument("-s", "--save-pmm", type=str, default=None, help="Path to file for saving the PMM state.")
    parser.add_argument("--save-loss", type=str, default=None, help="Path to file for saving loss information.")
    parser.add_argument("-e", "--epochs", type=int, default=10000, help="Number of cycles to run PMM algorithm for.")
    parser.add_argument("-L", "--parameters", type=str, default="1.0", help="Parameter values at which to predict energies.")
    parser.add_argument("--store-loss", type=int, default=100, help="Frequency for storing the computing loss.")
    parser.add_argument("--no-normalize", action="store_true", help="Runs the PMM without normalizing the sample data read from the input file.")
    parser.add_argument("-k", "--knum", type=int, default=None, help="The number of eigenvalues to write. Default is all.")
    parser.add_argument("-q", "--quiet", action="store_false", help="Sets quiet mode. Print output suppressed.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    args = parser.parse_args()
    return args

def _parse_config_and_pmm(pmm_name_str, config_str, config_file):
    # load config string from --config and config file from --config-file
    # overwrite config file with key=val pairs from config string if both given
    config_dict = parse.parse_kwargs(config_str)
    config_file_dict = parse.parse_config_dict(io.load_pmm_config(config_file)) if config_file is not None else {}
    pmm_kwargs = config_file_dict | config_dict
 
    # parse PMMClass from --pmm-name input and instantiate pmm
    PMMClass = parse.parse_pmm_string(pmm_name_str)
    pmm_instance = PMMClass(**pmm_kwargs)

    logger.debug(f"Parsed {pmm_name_str} -> {type(pmm_instance).__name__}")
    logger.debug(f"Parsed config -> " + 
                 ", ".join(f"{k}={v} ({type(v).__name__})" for k, v in pmm_instance._init_kwargs.items()))
    return pmm_instance

def _load_data_from_input_file(input_file):
    if input_file.endswith(".h5"):
        sample_Ls, sample_energies, _, _ = io.load_energies_from_h5(input_file)
        logger.debug(f"Loaded energy data from HDF5 file.")
    else:
        sample_Ls, sample_energies, _ = io.load_energies_from_dat(input_file)
        logger.debug(f"Loaded energy data from .dat-type file.")
    return sample_Ls, sample_energies

def _parse_parameter_values(parameter_str):
    predict_Ls = parse.parse_parameter_values(parameter_str)
    logger.debug(f"Parameters {parameter_str} parsed -> {type(predict_Ls).__name__} "
                 f"with min={min(predict_Ls)}, max={max(predict_Ls)}, len={len(predict_Ls)}")
    return predict_Ls 

def _sample_train_predict_notnormed(args, pmm_instance, sample_Ls, sample_energies, predict_Ls):
    # sample pmm with data loaded from file
    logger.info("Sampling pmm with (un-normalized) energies loaded from file.")
    pmm_instance.sample_energies(sample_Ls, sample_energies)
    # train pmm with sampled data
    logger.info(f"Training pmm for {args.epochs} cycles and storing loss every {args.store_loss} cycles.")
    _, losses = pmm_instance.train_pmm(args.epochs, args.store_loss)
    # predict energies at given parameter values with trained PMM
    logger.info("Predicting energies of trained PMM at given parameter values.")
    predicted_energies = pmm_instance.predict_energies(predict_Ls)
    return losses, predicted_energies

def _sample_train_predict_normed(args, pmm_instance, sample_Ls, sample_energies, predict_Ls):
    # sample pmm with data loaded from file after normalizing it
    logger.info("Sampling pmm with (normed) energies loaded from file.")
    logger.debug("Normalizing sample data before training PMM.")
    lmin, lmax, normed_sample_Ls = utils.normalize(sample_Ls)
    emin, emax, normed_sample_energies = utils.normalize(sample_energies)
    pmm_instance.sample_energies(normed_sample_Ls, normed_sample_energies)
    
    # train pmm with sampled data
    logger.info(f"Training pmm for {args.epochs} cycles and storing loss every {args.store_loss} cycles.")
    _, losses = pmm_instance.train_pmm(args.epochs, args.store_loss)

    # predict energies at given parameter values with trained PMM wrt normalization of sample set
    logger.debug("Normalizing prediction parameters with respect to sample parameters.")
    _, _, normed_predict_Ls = utils.normalize(predict_Ls, lmin, lmax)
    logger.info("Predicting energies of trained PMM.")
    normed_predicted_energies = pmm_instance.predict_energies(normed_predict_Ls)
    logger.debug(f"Denormalizing energy predictions with respect to sample energies.")
    predicted_energies = utils.denormalize(normed_predicted_energies, emin, emax)
    return losses, predicted_energies

def _get_metadata(args, pmm_instance):
    metadata = {"timestamp" : time.strftime("%A, %b %d, %Y %H:%M:%S"),
                "PMM" : args.pmm_name,
                "pmm_config" : ','.join(f"{k}={v}" for k, v in pmm_instance._init_kwargs.items()),
                "input_file" : args.input_file,
                "parameters" : args.parameters,
                "epochs" : args.epochs,
                "command" : ' '.join(sys.argv)
                }
    return metadata

def _write_energy_data(save_energies_path, predict_Ls, predicted_energies, metadata):
    if save_energies_path.endswith(".h5"):
        io.write_energies_to_h5(save_energies_path, predict_Ls, predicted_energies, metadata=metadata)
        logger.debug(f"Saved energy to HDF5 file.")
    else:
        io.write_energies_to_dat(save_energies_path, predict_Ls, predicted_energies, metadata=metadata)
        logger.debug(f"Saved energy to .dat-type file.")
  

def _print_results(args, predict_Ls, predicted_energies, losses):
    print(losses[-1])
    trnc = 9 # num of eigenvalues to print before skipping rest
    knum = args.knum if args.knum is not None else len(predicted_energies[0])
    for i, L in enumerate(predict_Ls):
        print(f"Spectrum at L = {L:.3f}")
        print(f"\t[ " + ", ".join(f"{eigval:.6f}" for eigval in predicted_energies[i][:trnc]), end="")
        
        if knum > trnc: 
            print(f", ..., {predicted_energies[i][-1]:.6f} ] ({len(predicted_energies[i])} energies total)")
        else: 
            print(" ]")

def main():
    # parse args, setup logging, and start timer
    args = _parse_args()
    _setup_logging(args.verbose)  
    start = time.time()            
    print(f"Input = {args.input_file}\tPMM = pmm.{args.pmm_name}\tEpochs = {args.epochs}")

    # parse config
    logger.info(f"Parsing config data and instantiating PMM.")
    pmm_instance = _parse_config_and_pmm(args.pmm_name, args.config, args.config_file) 
    logger.info(f"Parsing parameter values.") 
    predict_Ls = _parse_parameter_values(args.parameters)
    
    # load data from input_file
    logger.info(f"Loading energy data from input file.")
    sample_Ls, sample_energies = _load_data_from_input_file(args.input_file)
    
    # sample pmm with loaded energies, train pmm, and predict at given parameter values
    if args.no_normalize:
        # sample, train, and predict without normalizing sample set
        losses, predicted_energies = _sample_train_predict_notnormed(args, pmm_instance, sample_Ls, sample_energies, predict_Ls)
    else:
        # sample, train, and predict with normalizing sample set
        losses, predicted_energies = _sample_train_predict_normed(args, pmm_instance, sample_Ls, sample_energies, predict_Ls)
    if args.knum is not None: predicted_energies = predicted_energies[:, :args.knum] # truncate to given knum

    # save energy data, pmm state, and loss data if desired
    metadata = _get_metadata(args, pmm_instance)
    if args.save_energies:
        logger.info(f"Saving energy data to out file.")
        _write_energy_data(args.save_energies, predict_Ls, predicted_energies, metadata)

    if args.save_pmm:
        logger.info("Saving PMM state.")
        io.save_pmm_state(args.save_pmm)

    if args.save_loss:
        logger.info("Saving loss data.")
        io.save_loss(args.save_loss, losses, args.store_loss, metadata)

    # print data
    if args.quiet:
        _print_results(args, predict_Ls, predicted_energies, losses)
    end = time.time()
    # print total time elapsed
    print(f"Done.\nElapsed time: {end - start:.3f} seconds.")

if __name__=="__main__":
    main()
