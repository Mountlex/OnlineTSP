
from cycler import cycler
import argparse
import sys
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('--save', action='store_true')

    return parser



def legend(name, param):
    if "replan" in name:
        return f"Replan"
    elif "ignore" in name:
        return f"Ignore"
    elif "smart_trust" in name:
        return f"SmartTrust (α={param})"
    elif "delayed_trust" in name:
        return f"DelayedTrust (α={param})"
    elif "smart" in name:
        return f"SmartStart"
    else:
        return "unknown"

def plot(filename, save):
    f = filename.split(".")[0]
    sns.set_theme(style='ticks')
    df = pd.read_csv(filename)

    if 'sigma' in list(df):
        x_name = 'sigma'
    else:
        x_name = 'frac'

    df = df[~df['name'].str.contains("delayed")]
    df = df[df['param']!=0.05]

    df = df.round(3)
    df['cr'] = df['alg'] / df['opt'] 
    df['param'] = df[['name','param']].apply(lambda x: legend(*x),axis=1)

    if 'frac' in list(df):
        df['frac'] = 10 * df['frac']

    ax = sns.lineplot(data=df, x=x_name, y="cr", hue='param', style='param', markers=('frac' in list(df)), linewidth=2.5, markersize=8)
    handlers,_ = ax.get_legend_handles_labels()
 
    

    fig = plt.gcf()
    fig.set_dpi(600)
    fig.set_size_inches(7,4)
    plt.ylabel('Empirical competitive ratio')
    
    if 'sigma' in list(df):
        ax.legend(handlers,df['param'].unique(),ncol=2, loc="upper left")
        ax.set_xscale('symlog', base=10)
        plt.xlabel("Noise parameter σ")

        ax.set(yscale='log')
        plt.tight_layout()
        plt.savefig(f"{f}.pdf")

        ax.set(yscale='linear')
        plt.ylim(bottom=0.9, top=2.15)
        plt.tight_layout()
        plt.savefig(f"{f}_zoom.pdf")
    else:
        _,labels = ax.get_legend_handles_labels()
        for p,l in zip(ax.get_lines(), labels):
            if l == "Ignore" or l == "Replan" or l == "SmartStart":
                p.set(marker=None)

        for h,l in zip(handlers, labels):
            if l == "Ignore" or l == "Replan" or l == "SmartStart":
                h.set(marker=None)
        ax.legend(handlers,df['param'].unique(),ncol=2, loc="upper left")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(bottom=0.9, top=2.3)
        ax.invert_xaxis()
        plt.xlabel("Number of correctly predicted requests.")      
        plt.tight_layout()
        plt.savefig(f"{f}.pdf")

    



if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if os.path.exists(parsed_args.file):

        plot(parsed_args.file, parsed_args.save)
    else:
        print("Path not valid!")
