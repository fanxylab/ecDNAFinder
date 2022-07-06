#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy  as np
from joblib import Parallel, delayed
import pybedtools as bt
import re

from .EcUtilities import Utilities
from .EcVisual import Visal
from .EcAnnotate import Annotate

class FilterLinks():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.CHR= [str(i) for i in range(1,23)] + ['MT','X','Y']
        self.PLM= ['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv',
                    '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                    'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                    'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                    'SunTag-CRISPRi', 'V7-MC-HG-FA']
        self.Ref= self.CHR + self.PLM

    def _getinfo(self):
        self.outdir= self.arg.Update
        self.arg.outpre = '%s/%s'%(self.arg.Update, self.arg.updatepre)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getupmerge(self):
        self._getinfo()
        if self.arg.upmerge:
            self.UpMerge = self.arg.upmerge
        else:
            self.UpMerge = self.arg.outpre + '.UpMerge'
        return self

    def _hubs(self,_G):
        if (_G.shape[0] >=self.arg.minhubsize) &\
           (_G.length.max()>=self.arg.minhublen) &\
           (_G.Supportsum.sum()>=self.arg.minhubnum) &\
           (_G['#chrom'].iloc[0] in self.CHR):
            return 'Hubs'
        else:
            return ''

    def _linkmark(self, UpMerge):
        UpMerge = UpMerge.merge( UpMerge.groupby(['#chrom', 'start', 'end'])\
                                        .apply(lambda x : self._hubs(x))\
                                        .to_frame('Hubs').reset_index(),
                                 on=['#chrom', 'start', 'end'], how='left')

        UpMerge['Covermax'] = UpMerge.Covers.apply(lambda x:max(map(float, str(x).split(';'))))
        UpMerge['Depthmax'] = UpMerge.Depths.apply(lambda x:max(map(float, str(x).split(';'))))

        UpMerge.insert(8, 'LinkMark', '')
        UpMerge.loc[(UpMerge.BPNumax   >= self.arg.breakpiontnum), 'LinkMark'] += 'MultiBP;'
        UpMerge.loc[(UpMerge.Covermax  >= self.arg.maxcoverage),   'LinkMark'] += 'Cover;'
        UpMerge.loc[(UpMerge.Depthmax  >= self.arg.maxdepth),      'LinkMark'] += 'Depth;'
        UpMerge.loc[(UpMerge.Supportsum>= self.arg.minsupportnum), 'LinkMark'] += 'Support;'
        UpMerge.loc[(UpMerge.SIDnum    >= self.arg.minsidnum),     'LinkMark'] += 'SupportID;'
        UpMerge.loc[(UpMerge.LINKSLen  <= self.arg.maxlinklen),    'LinkMark'] += 'Len;'
        UpMerge.loc[(UpMerge.Hubs =='Hubs'), 'LinkMark'] += 'Hubs;'
        UpMerge.drop(['Covermax', 'Depthmax'], axis=1, inplace=True)
        return UpMerge

    def _hubs2(self,_G):
        if (_G.shape[0] >= self.arg.minhubsize and
            min(int(x) for x in _G['Length'].iloc[0].split(',')) >= self.arg.minhublen and
            min(int(x) for x in _G['SupportNum'].iloc[0].split(',')) >= self.arg.minhubnum and
            _G['#chrom'].iloc[0] in self.CHR):
            return 'Hubs'
        else:
            return ''

    def _marklink(self, UpMerge):
        UpMerge = UpMerge.merge( UpMerge.groupby(['Region'])\
                                        .apply(lambda x : self._hubs2(x))\
                                        .to_frame('Hubs').reset_index(),
                                 on=['Region'], how='left')


        UpMerge['Covermax'] = UpMerge.Covers.apply(lambda x:max(map(float, str(x).split(','))))
        UpMerge['Depthmax'] = UpMerge.Depths.apply(lambda x:max(map(float, str(x).split(','))))
        UpMerge['LinksLen'] = UpMerge.LinkLengths.apply(lambda x:max(map(int, str(x).split(','))))
        UpMerge.insert(7, 'LinkMark', '')
        #UpMerge.loc[(UpMerge.BPNumax   >= self.arg.breakpiontnum), 'LinkMark'] += 'MultiBP;'
        UpMerge.loc[(UpMerge.Covermax >= self.arg.maxcoverage), 'LinkMark'] += 'Cover;'
        UpMerge.loc[(UpMerge.Depthmax >= self.arg.maxdepth), 'LinkMark'] += 'Depth;'
        UpMerge.loc[(UpMerge.ReadNum >= self.arg.minsupportnum), 'LinkMark'] += 'Support;'
        UpMerge.loc[(UpMerge.SIDNum >= self.arg.minsidnum), 'LinkMark'] += 'SupportID;'
        #UpMerge['LinkMark'] += 'Len;'
        UpMerge.loc[(UpMerge.TotalLength  <= self.arg.maxlinklen),    'LinkMark'] += 'Len;' # will drop MYC EGFR
        UpMerge.loc[(UpMerge.Hubs =='Hubs'), 'LinkMark'] += 'Hubs;'
        UpMerge.drop(['Covermax', 'Depthmax', 'LinksLen'], axis=1, inplace=True)
        return UpMerge


    def FilterLink(self):
        self._getupmerge()
        UpMerge = pd.read_csv(self.UpMerge, sep='\t', dtype={'#chrom':str, 'ReadNum':int}, low_memory=False)
        UpMerge = self._marklink(UpMerge)
        Keep = UpMerge[(UpMerge.LinkMark.str.contains('Support;SupportID;Len;',regex=False))]['Region']
        UpMerge = UpMerge[(UpMerge.Region.isin(Keep))].sort_values(by=['Type','#chrom','Region'])
        UpMerge = Annotate(self.arg, self.log).bptrfannot(UpMerge)
        UpMerge.to_csv(self.arg.outpre + '.UpFilter', sep='\t', index=False)

        UpFilter = pd.read_csv(self.arg.outpre + '.UpFilter', sep='\t')
        Chr = UpFilter[~(UpFilter['#chrom'].isin(self.Ref))]['Region'].tolist()
        Null = UpFilter[UpFilter.TRF.isnull()]['Region'].tolist()
        print(Chr)
        print(Null)
        null_trf = UpFilter[UpFilter.TRF.isnull()]
        UpFilter = UpFilter[UpFilter.TRF.notnull()]
        null_trf = null_trf[null_trf.ReadNum > 10]
        print(null_trf[['#chrom', 'Region', 'ReadNum']])
        #print(UpFilter)
        TRF = UpFilter[UpFilter.TRF.str.contains('trf')]['Region'].tolist()
        #print(Chr)
        #print(Null)
        #print(TRF)
        #HUB = UpFilter[( (UpFilter.Hubs.str.contains('Hubs')) & (UpFilter.TRF.str.contains('trf')) )]['Region'].tolist()
        #UpFilter = UpFilter[~(UpFilter.Region.isin(Chr+TRF+HUB))]
        print(UpFilter[UpFilter.Region.isin(Null)])
        UpFilter = pd.concat([UpFilter[~(UpFilter.Region.isin(Chr+TRF))], null_trf], axis=0)
        UpFilter.to_csv(self.arg.outpre + '.UpFilterTRF', sep='\t',index=False)

    def FormatLink(self):
        self._getupmerge()

        UpMerge = pd.read_csv(self.UpMerge, sep='\t', dtype={'#chrom':str}, low_memory=False)
        UpMerge = self._linkmark(UpMerge)

        #Keep = UpMerge[(UpMerge.LinkMark.str.contains('Support;SupportID;Len;|Len;Hubs;',regex=True))]['LINKS']
        Keep = UpMerge[(UpMerge.LinkMark.str.contains('Support;SupportID;Len;',regex=False))]['LINKS']
        UpMerge = UpMerge[(UpMerge.LINKS.isin(Keep))].sort_values(by=['Type','#chrom','LINKS'])
        UpMerge = Annotate(self.arg, self.log).bptrfannot(UpMerge)
        UpMerge.to_csv(self.arg.outpre + '.UpFilter', sep='\t', index=False)

        UpFilter = pd.read_csv(self.arg.outpre + '.UpFilter', sep='\t')
        Chr = UpFilter[~(UpFilter['#chrom'].isin(self.Ref))]['LINKS'].tolist()
        TRF = UpFilter[( (UpFilter.BPLoci.str.contains('BP')) & (UpFilter.TRF.str.contains('trf')) )]['LINKS'].tolist()
        #HUB = UpFilter[( (UpFilter.Hubs.str.contains('Hubs')) & (UpFilter.TRF.str.contains('trf')) )]['LINKS'].tolist()
        #UpFilter = UpFilter[~(UpFilter.LINKS.isin(Chr+TRF+HUB))]
        UpFilter = UpFilter[~(UpFilter.LINKS.isin(Chr+TRF))]
        UpFilter.to_csv(self.arg.outpre + '.UpFilterTRF', sep='\t',index=False)
