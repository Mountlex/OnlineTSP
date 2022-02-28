
from cycler import cycler
import argparse
import sys
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    elif "smart" in name:
        return f"SmartStart"
    elif "pred" in name:
        return f"SmartTrust (α = {param})"
    else:
        return "unknown"

def plot(filename, save):
    sns.set_theme(style='ticks')

    x_name = 'sigma'

    df = pd.read_csv(filename)
    df = df.round(3)
    df['cr'] = df['alg'] / df['opt'] 
    df['param'] = df[['name','param']].apply(lambda x: legend(*x),axis=1)

    ax = sns.lineplot(data=df, x=x_name, y="cr", hue='param', style='param', markers=('round' in list(df)), linewidth=2.5, markersize=8)
    handlers,_ = ax.get_legend_handles_labels()
 
    ax.legend(handlers,df['param'].unique(),ncol=2, loc="upper left")
    ax.set(xscale='symlog')
    #ax.set(yscale='log')
    plt.ylim(bottom=0.9, top=4)
    plt.xlabel("Noise parameter")

    plt.ylabel('Empirical competitive ratio')
    plt.tight_layout()

    fig = plt.gcf()
    fig.set_dpi(600)
    fig.set_size_inches(7,4)

    if save:
        f = filename.split(".")[0]
        plt.savefig(f"{f}.pdf")
    else:
        plt.show()



if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if os.path.exists(parsed_args.file):

        plot(parsed_args.file, parsed_args.save)
    else:
        print("Path not valid!")
