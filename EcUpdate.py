#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy  as np
from joblib import Parallel, delayed

from .EcMerge import MergeReads
from .EcVisual import Visal

class UpdateCat():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.maxreadstwoends = self.arg.maxreadstwoends
        self.readsmergeways  = self.arg.readsmergeways

        self.outdir= self.arg.Update
        self.arg.outpre= '%s/%s'%(self.arg.Update, self.arg.updatepre)
        os.makedirs(self.outdir, exist_ok=True)

    def AllEcDNA(self, _info):
        self.log.CI('start merging all samples region.')
        Allbed = []
        for _n, _l in _info.iterrows():
            EMerge = '{0}/{1}/{1}.Keep'.format(self.arg.Search, _l.sampleid)
            if os.path.exists( EMerge ):
                Allbed.append( pd.read_csv(EMerge, sep='\t', header=0, dtype={'#chrom':str}, low_memory=False) )
            else:
                self.log.CW('cannot find the file: '+ EMerge)

        if Allbed:
            Allbed = pd.concat(Allbed, axis=0,sort=False)
            Allbed[['start', 'end', 'length']]  = Allbed[['start', 'end', 'length']].astype(int)
            Allbed['#chrom']  = Allbed['#chrom'].astype(str)
            Allbed['HTSites'] = Allbed['HTSites'].map(eval)
            Allbed.sort_values(by=['#chrom', 'start', 'end'], inplace=True)
            Allbed.to_csv(self.arg.outpre+'.Keep', sep='\t', index=False)
            MergeReads(self.arg, self.log).mergeReads(Allbed, Lplot=True)
        else:
            self.log.CW('cannot find the valid files.')

        self.log.CI('finish merging all samples region.')
