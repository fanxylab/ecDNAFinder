#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy  as np
from joblib import Parallel, delayed
import copy
import pybedtools as bt
import warnings
warnings.filterwarnings("ignore")


from EcMagiccube import LinkAnnot, BTAnnot, Transreg

class Annotate():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.maxreadstwoends = self.arg.maxreadstwoends
        self.readsmergeways  = self.arg.readsmergeways
        self.minoverlap  = self.arg.minoverlap
        self.minovertrf  = self.arg.minovertrf
        self.trfdistance = self.arg.trfdistance
        self.annotcol    = self.arg.annotcol
        self.annotbplen  = self.arg.annotbplen
        self._getgenebed()

    def _getgenebed(self):
        #gtfbed: '#chrom', 'start', 'end', 'gene_name', 'gene_biotype'
        self.gtfbed  = self.arg.gtfbed
        self.biotype = self.arg.biotype
        self.genecolr= ['#chrom', 'start', 'end']  + [self.annotcol , 'gene_biotype']
        self.genecoln= ['#chrom_g', 'start_g', 'end_g']  + [self.annotcol , 'gene_biotype']

        Gene = pd.read_csv(self.gtfbed, sep='\t', header=0)
        Gene = Gene[self.genecolr][ (Gene.gene_biotype.isin(self.biotype)) ]
        Gene['#chrom'] = Gene['#chrom'].astype(str)
        Gene.sort_values(by=['#chrom', 'start', 'end'], ascending=[True, True,  True], inplace=True)
        Gene.columns = self.genecoln
        self.gbed = bt.BedTool.from_dataframe(Gene)
        return Gene

    def _gettrfbed(self):
        COL = ['#chrom_t', 'start_t', 'end_t']
        TRF = pd.read_csv(self.arg.simplerepeat, sep='\t', header=None, names=COL + ['trf'])
        TRF.drop('trf', axis=1, inplace=True)
        TRF['#chrom_t'] = TRF['#chrom_t'].astype(str)
        TRF = TRF[(TRF['#chrom_t'].str.len()<6)]
        TRF = bt.BedTool.from_dataframe(TRF)
        TRF = TRF.sort().merge()
        return COL, TRF

    def geneannotc(self, _anno):
        _anno=_anno.copy()
        _genbed = self._getgenebed().values
        _anno[['gene_name', 'gene_biotype']] = \
                _anno[['#chrom', 'start', 'end']]\
                .apply(lambda x: LinkAnnot( Transreg(x.tolist(), _l=self.annotbplen),
                                            _genbed, minover=self.minoverlap), axis=1)\
                .tolist()
        _anno[['gene_name', 'gene_biotype']] = _anno[['gene_name', 'gene_biotype']].replace({'':'.'})
        return _anno

    def geneannotb(self, _anno):
        _anno=_anno.copy()
        _genbed = self._getgenebed().values
        _anno[['gene_name', 'gene_biotype']] = \
                _anno[['#chrom','BPs']]\
                .apply(lambda x: LinkAnnot(x.tolist(), _genbed, minover=self.minoverlap), axis=1)\
                .tolist()
        _anno[['gene_name', 'gene_biotype']] = _anno[['gene_name', 'gene_biotype']].replace({'':'.'})
        return _anno

    def geneannota(self, _inbed):
        _genbed = self._getgenebed()
        _tcol   = _genbed.columns.tolist()
        gbed    = bt.BedTool.from_dataframe(_genbed)

        _bcol= ['#chrom', 'start', 'end']
        bbed = _inbed[_bcol].drop_duplicates(keep='first')
        bbed['#chrom'] = bbed['#chrom'].astype(str)
        bbed = bt.BedTool.from_dataframe(bbed)

        COLs = _bcol + _tcol
        bbed  = bbed.intersect(gbed, s=False, S=False, wa=True, wb=True)\
                    .to_dataframe(disable_auto_names=True, header=None, names=COLs)
        del gbed

        COL1 = _bcol + [self.annotcol, 'gene_biotype']
        bbed = bbed.infer_objects()
        bbed['#chrom'] = bbed['#chrom'].astype(str)
        bbed['len_m']  = bbed[['end', 'end_g']].min(1) - bbed[['start', 'start_g']].max(1)
        bbed = bbed.loc[(bbed.len_m >= self.minoverlap), COL1]

        kbed = bbed.groupby(by=_bcol)
        bbed = pd.concat([ kbed[ self.annotcol].apply(lambda x:x.astype(str).str.cat(sep=';')).to_frame('gene_name'),
                           kbed['gene_biotype'].apply(lambda x:x.astype(str).str.cat(sep=';')).to_frame('gene_biotype')],
                        axis=1).reset_index()

        bbed = _inbed.merge(bbed, on=_bcol, how='left').copy()
        bbed[['gene_name', 'gene_biotype']] = bbed[['gene_name', 'gene_biotype']].fillna('.')
        return bbed

    def geneannote(self, chrom, start, end):
        if start < end:
            dbed = bt.BedTool(f'{chrom}\t{start}\t{end}', from_string=True)
        else:
            dbed = bt.BedTool(f'{chrom}\t{end}\t{start}', from_string=True)
        COLs = ['#chrom', 'start', 'end', 'chr', 's', 'e', 'name', 'biotype']
        bbed = dbed.intersect(self.gbed, s=False, S=False, wa=True, wb=True)\
            .to_dataframe(disable_auto_names=True, header=None, names=COLs)
        if not len(bbed.index):
            return '', ''
        return ','.join(bbed['name']), ','.join(bbed['biotype'])

    def bptrfannot(self, _inbed):
        _tcol, _trf = self._gettrfbed()
        _bcol = ['#chrom', 'start', 'end']
        bbed = _inbed[_bcol].drop_duplicates(keep='first')
        bbed['#chrom'] = bbed['#chrom'].astype(str)
        bbed['nstart'] = bbed['start']
        bbed['nend'] = bbed['end']
        #bbed['strand'] = '+'
        for i, q in bbed.iterrows():
            if q['start'] > q['end']:
                bbed['nstart'][i], bbed['nend'][i] = q['end'], q['start']
             #   bbed['strand'] = '-'
        # only annotate the bp loci retion
        #K = tbed.apply(lambda x: BTAnnot(x.tolist(), _trf.values), axis=1)
        nbcol = ['#chrom', 'nstart', 'nend', 'start', 'end']
        bbed = bbed[nbcol]

        _COLs = nbcol + _tcol
        bbed = bt.BedTool.from_dataframe(bbed)
        bbed  = bbed.intersect(_trf, s=False, S=False, wa=True, wb=True)\
                    .to_dataframe(disable_auto_names=True, header=None, names=_COLs)

        bbed = bbed.infer_objects()
        bbed['#chrom'] = bbed['#chrom'].astype(str)
        bbed['len_m']  = bbed[['nend', 'end_t']].min(1) - bbed[['nstart', 'start_t']].max(1)
        bbed = bbed[(bbed.len_m >= self.minovertrf)]
        bbed['TRF']  = 'Over'

        bbed['start_t']  -= self.trfdistance
        bbed['end_t']    += self.trfdistance
        bbed.loc[( bbed.start.between(bbed.start_t, bbed.end_t)),'TRF'] +=';Htrf'
        bbed.loc[( bbed.end.between(bbed.start_t, bbed.end_t) ), 'TRF'] +=';Ttrf'

        bbed = bbed[_bcol + ['TRF']]\
                    .groupby(by=_bcol, sort=False)['TRF']\
                    .apply(lambda x: ';'.join(sorted(set(';'.join(x.unique()).split(';')))))\
                    .to_frame('TRF').reset_index()
        bbed.to_csv('after.txt', sep='\t')

        bbed = _inbed.merge(bbed, on=_bcol, how='left')
        return bbed
