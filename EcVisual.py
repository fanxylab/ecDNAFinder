#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import numpy  as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
import matplotlib.pyplot as plt
import seaborn as sns

class Visal():
    def __init__(self):
        pass
    
    def query_length(self, _indf, out, X='query_length', Dup='query_name', log=False, title=''):
        if not _indf.empty:
            indef = _indf.copy()
            if Dup:
                indef = indef[[Dup, X]].drop_duplicates(keep='first')

            indef[X] = indef[X].astype(int)
            dp = sns.displot(data=indef, x=X, kde=True, log_scale=log)
            dp.set_xticklabels(rotation=270)

            if title:
                plt.title(title)

            plt.tight_layout()
            plt.savefig( out )
            plt.close()

    def clustmap(self, _indf, out, Trans=False):
        linewidths= 0 if min(_indf.shape) > 60  else 0.01
        hm = sns.clustermap(_indf,
                    method='complete',
                    metric='euclidean',
                    z_score=None,
                    #figsize=figsize,
                    linewidths=linewidths,
                    cmap="viridis_r",
                    cbar_pos=(0.02, 0.83, 0.03, 0.11)
                    #center=0,
                    #fmt='.2f',
                    #square=True, 
                    #cbar=True,
                    #yticklabels=Xa,
                    #xticklabels=Xa,
                    #vmin=-1.1,
                    #max=1.1,
                    )
        hm.savefig(out)
        #hm.fig.subplots_adjust(right=.2, top=.3, bottom=.2)
        plt.close()

    def lineplt(self, df, out):
        '''
        g = sns.relplot(
            data=df,
            x="Bins", y="log2_count", col="chrom", hue="SID",
            kind="line", palette="viridis_r", linewidth=1, zorder=5,
            col_wrap=5, height=5, aspect=1.5, legend=True,
        )
        g.savefig(out)
        #hm.fig.subplots_adjust(right=.2, top=.3, bottom=.2)
        '''
        g = sns.FacetGrid(df, col="GrpC", row="SID", sharex=False)
        g.map_dataframe(sns.scatterplot, x="Bins",  y="log2_count")
        #g.set_axis_labels("Total bill", "Count")
        g.savefig(out)
        plt.close()

    def Lmplot(self, out, title='', **kargs):
        g = sns.lmplot(**kargs)
        #g.set_xticklabels(rotation=270)
        g.set(xlim=(0, 30))
        g.set(ylim=(0, 200))
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.savefig( out )
        plt.close()

    def Pdistribut(self, _indf, out, X='query_length', Dup=[], Bins=[], logx=True, logy=True, title='', xlim=None):
        if not _indf.empty:
            indef = _indf.copy()
            if Dup:
                indef = indef[Dup +[X]].drop_duplicates(keep='first')
            indef[X] = indef[X].astype(float)
            if xlim:
                indef = indef[((indef[X]>=xlim[0]) & (indef[X]<=xlim[1])) ]

            if Bins:
                dp = sns.displot(data=indef, x=X, bins=Bins, kde=False, log_scale=logx)     
            else:
                dp = sns.displot(data=indef, x=X, kde=True, log_scale=logx)
            dp.set_xticklabels(rotation=270)

            if title:
                plt.title(title)
            if logy:
                plt.yscale('log')

            plt.tight_layout()
            plt.savefig( out )
            plt.close()

    def GCdistribut(self, _indf, out, X='query_length', Dup=[], log=False, title=''):
        if not _indf.empty:
            indef = _indf.copy()
            if Dup:
                indef = indef[Dup +[X]].drop_duplicates(keep='first')
            indef[X] = indef[X].astype(float)
            dp = sns.displot(data=indef, x=X, kde=True, log_scale=log)
            dp.set_xticklabels(rotation=270)
            if title:
                plt.title(title)
            plt.tight_layout()
            plt.savefig( out )
            plt.close()

    def GCcounts(self, X='mGC', y='counts', data= pd.DataFrame(), out='./gcounts.pdf', Dup=[], title=''):
        if not data.empty:
            indef = data.copy()
            if Dup:
                indef = indef[Dup +[X]].drop_duplicates(keep='first')
            dp = sns.lmplot(x=X, y=y, data=data, 
                            scatter_kws={"s": 0.1, 'color': 'red'}, 
                            line_kws={'color':'blue', 'linewidth':1})
            dp.set_xticklabels(rotation=270)
            if title:
                plt.title(title)
            plt.tight_layout()
            plt.savefig( out )
            plt.close()
