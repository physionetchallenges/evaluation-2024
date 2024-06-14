#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py -d labels -o outputs -s scores.csv
#
# where 'labels' is a folder containing files with the labels for the data, 'outputs' is a folder containing files with the outputs
# from your models, and 'scores.csv' (optional) is a collection of scores for the model outputs.
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
    description = 'Evaluate the Challenge models.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--input_folder', type=str, required=True)
    parser.add_argument('-o', '--output_folder', type=str, required=True)
    parser.add_argument('-x', '--extra_scores', action='store_true')
    parser.add_argument('-s', '--score_file', type=str, required=False)
    return parser

# Evaluate the models.
def evaluate_model(input_folder, output_folder, extra_scores=False):
    # Find the records.
    records = find_records(input_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No records found.')

    # Compute the digitization metrics.
    records_completed_digitization = list()
    snr = dict()
    snr_median = dict()
    ks_metric = dict()
    asci_metric = dict()
    weighted_absolute_difference_metric = dict()

    # Iterate over the records.
    for record in records:
        # Load the signals, if available.
        input_record = os.path.join(input_folder, record)
        input_signal, input_fields = load_signals(input_record)

        if input_signal is not None:
            input_channels = input_fields['sig_name']
            input_num_channels = input_fields['n_sig']
            input_num_samples = input_fields['sig_len']
            input_sampling_frequency = input_fields['fs']
            input_units = input_fields['units']

            output_record = os.path.join(output_folder, record)
            output_signal, output_fields = load_signals(output_record)

            if output_signal is not None:
                output_channels = output_fields['sig_name']
                output_num_channels = output_fields['n_sig']
                output_num_samples = output_fields['sig_len']
                output_sampling_frequency = output_fields['fs']
                output_units = output_fields['units']

                records_completed_digitization.append(record)

                # Check that the input and output signals match as expected.
                assert(input_sampling_frequency == output_sampling_frequency)
                assert(input_units == output_units)

                # Reorder the channels in the output signal to match the channels in the input signal.
                output_signal = reorder_signal(output_signal, output_channels, input_channels)

                # Trim or pad the channels in the output signal to match the channels in the input signal.
                output_signal = trim_signal(output_signal, input_num_samples)

                # Replace the samples with NaN values in the output signal with zeros.
                output_signal[np.isnan(output_signal)] = 0

            else:
                output_signal = np.zeros_like(input_signal)

            # Compute the digitization metrics.
            channels = input_channels
            num_channels = input_num_channels
            sampling_frequency = input_sampling_frequency

            for j, channel in enumerate(channels):
                value = compute_snr(input_signal[:, j], output_signal[:, j])
                snr[(record, channel)] = value

                if extra_scores:
                    value = compute_snr_median(input_signal[:, j], output_signal[:, j])
                    snr_median[(record, channel)] = value

                    value = compute_ks_metric(input_signal[:, j], output_signal[:, j])
                    ks_metric[(record, channel)] = value

                    value = compute_asci_metric(input_signal[:, j], output_signal[:, j])
                    asci_metric[(record, channel)] = value

                    value = compute_weighted_absolute_difference(input_signal[:, j], output_signal[:, j], sampling_frequency)
                    weighted_absolute_difference_metric[(record, channel)] = value

    # Compute the metrics.
    if len(records_completed_digitization) > 0:
        snr = np.array(list(snr.values()))
        if not np.all(np.isnan(snr)):
            mean_snr = np.nanmean(snr)
        else:
            mean_snr = float('nan')

        if extra_scores:
            snr_median = np.array(list(snr_median.values()))
            if not np.all(np.isnan(snr_median)):
                mean_snr_median = np.nanmean(snr_median)
            else:
                mean_snr_median = float('nan')

            ks_metric = np.array(list(ks_metric.values()))
            if not np.all(np.isnan(ks_metric)):
                mean_ks_metric = np.nanmean(ks_metric)
            else:
                mean_ks_metric = float('nan')

            asci_metric = np.array(list(asci_metric.values()))
            if not np.all(np.isnan(asci_metric)):
                mean_asci_metric = np.nanmean(asci_metric)
            else:
                mean_asci_metric = float('nan')

            weighted_absolute_difference_metric = np.array(list(weighted_absolute_difference_metric.values()))
            if not np.all(np.isnan(weighted_absolute_difference_metric)):
                mean_weighted_absolute_difference_metric = np.nanmean(weighted_absolute_difference_metric)
            else:
                mean_weighted_absolute_difference_metric = float('nan')
        else:
            mean_snr_median = float('nan')
            mean_ks_metric = float('nan')
            mean_asci_metric = float('nan')
            mean_weighted_absolute_difference_metric = float('nan')

    else:
        mean_snr = float('nan')
        mean_snr_median = float('nan')
        mean_ks_metric = float('nan')
        mean_asci_metric = float('nan')
        mean_weighted_absolute_difference_metric = float('nan')

    # Compute the classification metrics.
    records_completed_classification = list()
    input_labels = list()
    output_labels = list()

    # Iterate over the records.
    for record in records:
        # Load the labels, if available.
        input_record = os.path.join(input_folder, record)
        try:
            input_label = load_labels(input_record)
        except:
            input_label = list()

        if any(label for label in input_label):
            output_record = os.path.join(output_folder, record)
            try:
                output_label = load_labels(output_record)
            except:
                output_label = list()

            if any(label for label in output_label):
                records_completed_classification.append(record)

            input_labels.append(input_label)
            output_labels.append(output_label)

    # Compute the metrics.
    if len(records_completed_classification) > 0:
        f_measure, _, _ = compute_f_measure(input_labels, output_labels)
    else:
        f_measure = float('nan')

    # Return the results.
    return mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, mean_weighted_absolute_difference_metric, f_measure

# Run the code.
def run(args):
    # Compute the scores for the model outputs.
    scores = evaluate_model(args.input_folder, args.output_folder, args.extra_scores)

    # Unpack the scores.
    snr, snr_median, ks_metric, asci_metric, mean_weighted_absolute_difference_metric, f_measure = scores

    # Construct a string with scores.
    if not args.extra_scores:
        output_string = \
            f'SNR: {snr:.3f}\n' + \
            f'F-measure: {f_measure:.3f}\n'
    else:
        output_string = \
            f'SNR: {snr:.3f}\n' + \
            f'SNR median: {snr_median:.3f}\n' \
            f'KS metric: {ks_metric:.3f}\n' + \
            f'ASCI metric: {asci_metric:.3f}\n' \
            f'Weighted absolute difference metric: {mean_weighted_absolute_difference_metric:.3f}\n' \
            f'F-measure: {f_measure:.3f}\n'

    # Output the scores to screen and/or a file.
    if args.score_file:
        save_text(args.score_file, output_string)
    else:
        print(output_string)

if __name__ == '__main__':
    run(get_parser().parse_args(sys.argv[1:]))