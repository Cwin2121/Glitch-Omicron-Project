#!/usr/bin/env python3.8
# coding: utf-8

#usage: python getomicron-site.py --start [GPS START TIME] --end [GPS END TIME] --ifo IFO --channel CHANNEL [--snr SNR] [--plot] [--verbose]
# Example of standard configuration:
    # python getomicron-site.py --start 1439800926 --end 1439811926 --ifo H1 --channel LSC_REFL_A_LF_OUT_DQ --snr 6 --plot --verbose

# Imports

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import numpy.ma as ma
import os
import sys
import argparse
import matplotlib.cm as cm
import pandas as pd
import glob

from gwpy.timeseries import TimeSeries
from gwpy.spectrogram import Spectrogram
from gwpy.astro import range_timeseries
from gwpy.table import EventTable

# For debugging only
#np.set_printoptions(threshold=np.inf)

# Functions

def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='count', help='Verbose mode')
    parser.add_argument('--plot', action='count', help='Plots the triggers')
    parser.add_argument('--ifo', help='Interferometer (H1 or L1)', required=True)
    parser.add_argument('--channel', help=' Channel name. For example GDS_CALIB_STRAIN or LSC_REFL_A_LF_OUT_DQ', required=True)
    parser.add_argument('--snr',  type=float, help='Minimum SNR of selected omicron triggers. (Positive float, default = 5.0)', default=5.0, required=False)
    parser.add_argument('--start',  type=int, help='GPS start time.', required=True)
    parser.add_argument('--end',  type=int, help='GPS end time', required=True)
    return parser.parse_args()

def make_omicron_table():
    cumulative_selected_table = pd.DataFrame(columns=features).astype(float)

    for omicron_file in glob.glob('/home/detchar/triggers/'+args.ifo+'/'+args.channel+'_OMICRON/*/'+args.ifo+'-'+args.channel+'_OMICRON*.h5', recursive=True):

        range_min = int(omicron_file.split('-')[-2])
        range_max = int(omicron_file.split('-')[-1].split('.')[0])+range_min
        if np.logical_and(range_max >= args.start, range_min < args.end):
            table = EventTable.read(omicron_file, path='triggers').to_pandas()
            time_range = np.logical_and(table['tstart'] >= args.start, table['tend'] < args.end)
            time_selected_table = table[time_range]
            selected_table = time_selected_table[time_selected_table['snr'] >= args.snr]
            cumulative_selected_table = pd.concat([cumulative_selected_table, selected_table], ignore_index=True)

    cumulative_selected_table.sort_values(by='time')

    return cumulative_selected_table

def write_omicron_table(omicron_table):
    if args.verbose:
        print('Writing the omicron table for segment ['+str(args.start)+','+str(args.end) +'] seconds.')

    os.makedirs(currentDirectory +'/Omicron/', exist_ok=True)
    filename = currentDirectory +'/Omicron/table-'+args.channel+'-snr_'+str(args.snr).replace('.','d')+'-start_'+str(args.start)+'-end_'+str(args.end)+'.csv'   
    omicron_table.to_csv(filename, index=False)
    return


def plot_omicron_triggers(omicron_table):
    if args.verbose:
        print('Plotting the omicron triggers of segment ['+str(args.start)+','+str(args.end) +'] seconds.')

    legend_elements = []
    legend_labels = []
    trigger_colors = {args.snr:'green', args.snr+8:'magenta', args.snr+16:'blue', args.snr+32: 'red'}
       
    for color in trigger_colors:
        snr_range = omicron_table['snr'] >= color
        time_in_range  = omicron_table['time'][snr_range]
        snr_in_range = omicron_table['snr'][snr_range]
        time_in_range_maxerr  = omicron_table['tend'][snr_range] - omicron_table['time'][snr_range]
        time_in_range_minerr  = omicron_table['time'][snr_range] - omicron_table['tstart'][snr_range]        
        plt.errorbar(time_in_range, snr_in_range, None, (time_in_range_minerr, time_in_range_maxerr), capsize=2, ls='none', color=trigger_colors[color], marker='.',elinewidth=1)
        legend_elements.append(plt.scatter([0],[0], marker='.',color=trigger_colors[color]))
        legend_labels.append(r'SNR $\geq$'+ str(color))

    ax = plt.gca()
    ax.set_xlabel('Time [seconds] from ' + str(args.start),fontsize=11)
    ax.set_xlim(args.start, args.end)
    ax.set_xticks(np.linspace(args.start, args.end, 9))
    ax.set_xticklabels([s.replace('.0','') for s in map(str,np.linspace(0, args.end - args.start, 9))],fontsize=11)
    ax.set_title('Omicron triggers ['+str(args.start)+','+str(args.end) +'] (s)',fontsize=9)

    ax.legend(legend_elements,legend_labels, loc = 'upper right')#, title="Omicron triggers")

    if snr_in_range.any():
        ax.set_ylim(4,10**np.ceil(np.log10(np.max(snr_in_range)))+1)
    else:
         ax.set_ylim(4,10**4+1)   
    ax.set_yscale('log')
    ax.set_ylabel('SNR [%s]' %(args.channel),fontsize=11)
    ax.grid(visible=True, which='major', axis='both',linestyle='--',color='k', linewidth=0.5)

    os.makedirs(currentDirectory +'/Omicron/', exist_ok=True)    
    filename = currentDirectory +'/Omicron/plot-'+args.channel+'-snr_'+str(args.snr).replace('.','d')+'-start_'+str(args.start)+'-end_'+str(args.end)+'.png'

    plt.savefig(filename,dpi=300)
    plt.close()

    return

def main():

    global currentDirectory
    currentDirectory = os.getcwd()

    global args
    args = argsparser()

    global features
    features = ['time','frequency','tstart','tend','fstart','fend','snr','q','amplitude','phase']

    if args.verbose:
        print('Starting!')       

    omicron_table = make_omicron_table()
    write_omicron_table(omicron_table)

    if args.plot:
        plot_omicron_triggers(omicron_table)

    if args.verbose:
        print('Done!')       

    sys.exit()

if __name__ == "__main__":
    main()

