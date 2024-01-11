#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py -d labels -o outputs -s scores.csv
#
# where 'labels' is a folder containing files with the labels, 'outputs' is a folder containing files with the outputs from your
# model, and 'scores.csv' (optional) is a collection of scores for the model outputs.
#
# Each label or output file must have the format described on the Challenge webpage. The scores for the algorithm outputs are also
# described on the Challenge webpage.

import argparse
import numpy as np
import os
import os.path
import sys

from helper_code import *

# Parse arguments.
def get_parser():
    description = 'Evaluate the Challenge model(s).'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--label_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-s', '--score_file', type=str, required=False)
    return parser

# Evaluate the models.
def evaluate_model(label_folder, output_folder):
    # Get the records.
    records = find_records(label_folder)

    # Load the signals and diagnoses, when available, from the header files.
    snrs = list()
    label_dxs = list()
    output_dxs = list()

    # Iterate over records.
    for record in records:
        # Load signals, if available, and compute SNR.
        label_record = os.path.join(label_folder, record)
        output_record = os.path.join(output_folder, record)

        label_signal, label_fields = load_signal(label_record)
        output_signal, output_fields = load_signal(output_record)

        label_channels = label_fields['sig_name']
        output_channels = output_fields['sig_name']
        label_num_samples = label_fields['sig_len']
        output_num_samples = output_fields['sig_len']

        ###
        ### TO-DO: Perform checks, such as sampling frequency, units, etc.
        ###

        output_signal = reorder_signal(output_signal, output_channels, label_channels)
        output_signal = trim_signal(output_signal, label_num_samples)

        snr = compute_snr(label_signal, output_signal)
        snrs.append(snr)

        # Load diagnoses, if available.
        label_header_file = get_header_file(label_record)
        label_header = load_text(label_header_file)
        label_dx = get_diagnosis(label_header)
        label_dxs.append(label_dx)

        output_header_file = get_header_file(output_record)
        output_header = load_text(output_header_file)
        output_dx = get_diagnosis(output_header)
        output_dxs.append(output_dx)

    # Summarize results.

    ###
    ### TO-DO: Decide how to combine SNRs, especially if some signals are missing.
    ###

    if all(snr is not None for snr in snrs):
        snr = np.mean(snrs)
    else:
        snr = None

    f_measure, _, _ = compute_f_measure(label_dxs, output_dxs)

    # Return the results.
    return snr, f_measure

# Run the code.
def run(args):
    # Compute the scores for the model outputs.
    scores = evaluate_model(args.label_folder, args.output_folder)

    # Unpack the scores.
    snr, f_measure = scores

    # Construct a string with scores.
    output_string = \
        f'SNR: {snr:.3f}\n' + \
        f'F-measure: {f_measure:.3f}\n'

    # Output the scores to screen and/or a file.
    if args.score_file:
        save_text(args.score_file, output_string)
    else:
        print(output_string)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))