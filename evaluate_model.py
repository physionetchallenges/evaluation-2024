#!/usr/bin/env python

# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.

# This file contains functions for evaluating models for the Challenge. You can run it as follows:
#
#   python evaluate_model.py -d data -o outputs -s scores.csv
#
# where 'data' is a folder containing files with the reference signals and labels for the data, 'outputs' is a folder containing
# files with the outputs from your models, and 'scores.csv' (optional) is a collection of scores for the model outputs.
#
# Each data or output file must have the format described on the Challenge webpage. The scores for the algorithm outputs are also
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
    parser.add_argument('-d', '--folder_ref', type=str, required=True)
    parser.add_argument('-o', '--folder_est', type=str, required=True)
    parser.add_argument('-n', '--no_shift', action='store_true')
    parser.add_argument('-x', '--extra_scores', action='store_true')
    parser.add_argument('-s', '--score_file', type=str, required=False)
    return parser

# Evaluate the models.
def evaluate_model(folder_ref, folder_est, no_shift=False, extra_scores=False):
    # Find the records.
    records = find_records(folder_ref)
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
        record_ref = os.path.join(folder_ref, record)
        signal_ref, fields_ref = load_signals(record_ref)

        if signal_ref is not None:
            channels_ref = fields_ref['sig_name']
            num_channels_ref = fields_ref['n_sig']
            num_samples_ref = fields_ref['sig_len']
            sampling_frequency_ref = fields_ref['fs']
            units_ref = fields_ref['units']

            record_est = os.path.join(folder_est, record)
            signal_est, fields_est = load_signals(record_est)

            if signal_est is not None:
                channels_est = fields_est['sig_name']
                num_channels_est = fields_est['n_sig']
                num_samples_est = fields_est['sig_len']
                sampling_frequency_est = fields_est['fs']
                units_est = fields_est['units']

                records_completed_digitization.append(record)

                # Check that the reference and and digitized signals match as expected.
                assert(sampling_frequency_ref == sampling_frequency_est)
                assert(units_ref == units_est)

                # Check that the units for all of the channels are mV.
                assert(len(set(units_ref)) == 1 and sorted(set(units_ref))[0] == 'mV')

                # Reorder the channels in the digitzed signal to match the channels in the reference signal.
                signal_est = reorder_signal(signal_est, channels_est, channels_ref)

            else:
                signal_est = np.nan*np.ones(np.shape(signal_ref))

            # Compute the metrics.
            channels = channels_ref
            num_channels = num_channels_ref
            sampling_frequency = sampling_frequency_ref

            # Set limits on how far the signal can be shifted, and the number of quantization levels when shifting the signals.
            max_hz_shift = np.round(0.5*sampling_frequency)
            max_vt_shift = 1.0
            num_quant_levels = 2**8

            # Shift the digitied signals to better align with the reference signals.
            signal_ref_collection = list()
            signal_est_collection = list()
            for j, channel in enumerate(channels):
                signal_ref_collection.append(signal_ref[:, j])
                # Align the signals.
                if not no_shift:
                    signal_shifted, shift_hz, shift_vt = align_signals(signal_ref[:, j], signal_est[:, j], num_quant_levels=num_quant_levels)
                    if abs(shift_hz) <= max_hz_shift and abs(shift_vt) <= max_vt_shift:
                        signal_est_collection.append(signal_shifted)
                    else:
                        signal_est_collection.append(signal_est[:, j])
                else:
                    signal_est_collection.append(signal_est[:, j])

            # Compute the SNRs and, optionally, additional metrics.
            for j, channel in enumerate(channels):
                value, p_signal, p_noise = compute_snr(signal_ref_collection[j], signal_est_collection[j])
                snr[(record, channel)] = value

                if extra_scores:
                    value = compute_snr(signal_ref_collection[j], signal_est_collection[j], noise_median=True)
                    snr_median[(record, channel)] = value

                    value = compute_ks_metric(signal_ref_collection[j], signal_est_collection[j])
                    ks_metric[(record, channel)] = value

                    value = compute_asci_metric(signal_ref_collection[j], signal_est_collection[j])
                    asci_metric[(record, channel)] = value

                    value = compute_weighted_absolute_difference(signal_ref_collection[j], signal_est_collection[j], sampling_frequency)
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
    labels_ref = list()
    labels_est = list()

    # Iterate over the records.
    for record in records:
        # Load the labels, if available.
        record_ref = os.path.join(folder_ref, record)
        try:
            label_ref = load_labels(record_ref)
        except:
            label_ref = list()

        if any(label for label in label_ref):
            record_est = os.path.join(folder_est, record)
            try:
                label_est = load_labels(record_est)
            except:
                label_est = list()

            if any(label for label in label_est):
                records_completed_classification.append(record)

            labels_ref.append(label_ref)
            labels_est.append(label_est)

    # Compute the metrics.
    if len(records_completed_classification) > 0:
        f_measure, _, _ = compute_f_measure(labels_ref, labels_est)
    else:
        f_measure = float('nan')

    # Return the results.
    return mean_snr, mean_snr_median, mean_ks_metric, mean_asci_metric, mean_weighted_absolute_difference_metric, f_measure

# Run the code.
def run(args):
    # Compute the scores for the model outputs.
    scores = evaluate_model(args.folder_ref, args.folder_est, args.no_shift, args.extra_scores)

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
