#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy  as np
from joblib import Parallel, delayed
import pysam

from .EcVisual import Visal
from .EcUtilities import Utilities

class CheckBP():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.mapq = self.arg.minmapQ
        self.overfremin = self.arg.overfremin
        self.overlenmin = self.arg.overlenmin
        self.maxcheck = self.arg.maxchecksofttwoends
        self.bptnum = self.arg.bptnum
        self.bptotalrate = self.arg.bptotalrate

    def _getinfo(self, _info):
        self.info = _info
        self.inid = _info.sampleid
        self.inbam = '%s/%s.sorted.bam'%(self.arg.Bam, self.inid)
        self.outdir= '%s/%s'%(self.arg.Cheak, self.inid )
        self.arg.outpre= '%s/%s'%(self.outdir, self.inid)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _bamcigarsoft(self):
        samfile = pysam.AlignmentFile(self.inbam, "rb")
        SID   = self.inid
        sampd = []
        Head  = ['#chrom', 'start', 'end', 'SID', 'length', 'forword', 'query_name', 'query_length', 
                 'cigarreg', 'alignment_qlen', 'mapping_quality', 'flag']

        for read in samfile.fetch():
            if read.flag in [256, 272]:
                continue
            else:
                is_reverse = '-' if read.is_reverse else '+'
                Info = [read.reference_name, read.reference_start, read.reference_end, SID, read.reference_end -read.reference_start + 1,
                        is_reverse, read.query_name, read.infer_read_length(), (read.qstart, read.qend), read.query_alignment_length ,
                        read.mapping_quality,  read.flag]
                sampd.append(Info)
        samfile.close()

        sampd = pd.DataFrame(sampd, columns=Head)
        sampd = sampd.merge(sampd.groupby('query_name')['query_name'].size().reset_index(name='query_counts'),
                            on='query_name', how='outer')
        sampd.to_csv( self.arg.outpre + '.readsinfo.gz', sep='\t', index=False)
        return sampd

    def _getbeddb(self):
        #self.inbed = '{0}/{1}/{1}.chimeric.bed'.format(self.arg.Fetch, self.inid)
        #self.inbed = pd.read_csv( self.inbed, sep='\t', low_memory=False)
        self.inbed = self._bamcigarsoft()
        self.inbed[['start', 'end']]  = self.inbed[['start', 'end']].astype(int)
        self.inbed['#chrom']   = self.inbed['#chrom'].astype(str)
        #self.inbed['cigarreg'] = self.inbed.cigarreg.map(eval)
        self.inbed['fflag']    = 'DROP'
        self.inbed['raw_order'] = 1

        COLs  = ['#chrom', 'start', 'end',  'SID', 'length', 'forword', 'query_name', 'query_length',
                 'fflag', 'raw_order','query_counts',  'cigarreg']
        self.inbed = self.inbed[COLs]
        self.inbed.sort_values(by=['query_name', 'cigarreg', '#chrom', 'start', 'end' ], ascending=[True]*5, inplace=True)
        self.inbed['raw_order'] =  self.inbed.groupby(by=['SID', 'query_name'], sort=False)['raw_order'].apply(np.cumsum)

        self.inBP = pd.read_csv(self.arg.checkbed, sep='\t')
        self.inBP['#chrom'] = self.inBP['#chrom'].astype(str)
        self.inBP['start']  = self.inBP['start'].astype(int) -1
        self.inBP['end']    = self.inBP['end'].astype(int) -1
        self.inBP['lenght'] = self.inBP['end'] - self.inBP['start'] + 1

    def _getkeep(self):
        self.outdir= '%s/%s'%(self.arg.Cheak, 'BPState' )
        self.outpre= '%s/%s'%(self.outdir, 'All.plasmid')
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def Rmerge( self, intervals):
        """
        :param intervals: List[List[int]]
        :return: List[List[int]]
        """
        intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in intervals:
            if not merged or merged[-1][-1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][-1] = max(merged[-1][-1], interval[-1])
        merged = sum([i[1]-i[0] + 1 for i in merged ])
        return merged

    def BEDfilter(self, inbed):
        #####addinfor
        inbed['cigarreg'] = inbed.cigarreg.map(eval)
        inbed['start_o']  = inbed[['start', 'start_i']].max(1)
        inbed['end_o']    = inbed[['end', 'end_i']].min(1)
        inbed.sort_values(by=['SID', 'query_name', 'raw_order'], inplace=True)

        inbed.rename(columns={'Links_i':'Links'}, inplace=True)
        #####addstander
        GRPBy = ['Links', 'query_name', 'forword', 'end_i']
        inbed = inbed.merge(inbed.groupby(by=GRPBy)\
                                 .apply(lambda x: self.Rmerge(x[['start_o', 'end_o']].values.tolist()))\
                            .to_frame('OVERlen').reset_index(), on=GRPBy, how='left')
        inbed['OVERfre'] = (inbed['OVERlen']/inbed['lenght_i']).round(4)

        GRPBy = inbed.groupby(by=['Links', 'query_name'])
        GROUP = [ GRPBy['end_i'].unique().apply(lambda x:len(x)).to_frame('BP_count'),
                  GRPBy.apply(lambda x:  (x.end.max() - x.start.min() +1 )/x.plen_i.values[0] ).to_frame('BP_ratio'),
                  GRPBy['cigarreg'].first().str[0].to_frame('HeadSoft'),
                  GRPBy['cigarreg'].last().str[1].to_frame('TailSoft') ]
        GROUP = pd.concat(GROUP, axis=1, sort=False).reset_index()
        inbed = inbed.merge(GROUP, on=['Links', 'query_name'], how='left')

        inbed['HeadSoft'] = (inbed['HeadSoft']/inbed['query_length']).round(4)
        inbed['TailSoft'] = (1 - inbed['TailSoft']/inbed['query_length']).round(4)
        
        # add marker
        inbed.loc[((inbed.OVERfre  < self.overfremin) & (inbed.OVERlen < self.overlenmin)), 'fflag'] += ';OVERMIN'
        inbed.loc[(inbed.BP_count  < self.bptnum), 'fflag'] += ';BPLOWNUM'
        inbed.loc[(inbed.BP_ratio  < self.bptotalrate), 'fflag'] += ';BPLOWFRE'
        inbed.loc[((inbed.HeadSoft > self.maxcheck) | (inbed.TailSoft > self.maxcheck)),   'fflag'] += ';HEADTAIL'
        inbed.loc[(inbed.fflag=='DROP'), 'fflag'] = 'KEEP'
        return inbed

    def BPFetchBed(self, _inline):
        self._getinfo(_inline)
        self._getbeddb()
        intSect = Utilities(self.arg, self.log)\
                    .bedintersect(self.inbed, self.inBP, s=False, S=False, wa=True, wb=True)
        intSect.to_csv(self.arg.outpre + '.breakpoint.bed.txt', sep='\t', index=False)
        #intSect = pd.read_csv(self.arg.outpre + '.breakpoint.bed.txt', sep='\t')
        intSect = self.BEDfilter(intSect)
        intSect.to_csv(self.arg.outpre + '.breakpoint.Mark.txt', sep='\t', index=False)
        intSect = intSect[(intSect.fflag=='KEEP')]
        intSect.to_csv(self.arg.outpre + '.breakpoint.Keep.txt', sep='\t', index=False)

    def BPKEEP(self, _indf, Lplot=True):
        indf = _indf.groupby(by=['#chrom', 'Links', 'SID', ])['query_name']\
                    .unique().apply(lambda x:len(x)).to_frame('support_ID_num').reset_index()

        pvot = indf.pivot(index='#chrom', columns='SID', values='support_ID_num').fillna(0).astype(int)
        indf = indf.groupby(by=['#chrom', 'Links'])['support_ID_num'].sum().to_frame('support_num').reset_index()
        indf = indf.merge(pvot.reset_index(), on='#chrom').sort_values(by=['support_num', '#chrom'], ascending=[False, True])
        indf.to_csv(self.outpre+'.Keep.matrix', sep='\t', index=False)

        if Lplot:
            Visal().clustmap(pvot, self.outpre+'.Keep.matrix.pdf')
            Visal().clustmap(np.log2(pvot+1), self.outpre+'.Keep.matrix.log2.pdf')

    def BPStat(self, _info ):
        Parallel( n_jobs=self.arg.njob, verbose=1 )( delayed( self.BPFetchBed )(_l) for _n, _l in _info.iterrows())

        self.log.CI('start stating all samples region.')
        self._getkeep()
        BPKEEP = []
        for _n, _l in _info.iterrows():
            EMerge = '{0}/{1}/{1}.breakpoint.Keep.txt'.format(self.arg.Cheak, _l.sampleid)
            if os.path.exists( EMerge ):
                BPKEEP.append( pd.read_csv(EMerge, sep='\t', header=0) )
            else:
                self.log.CW('cannot find the file: '+ EMerge)
        if BPKEEP:
            BPKEEP = pd.concat(BPKEEP, axis=0,sort=False)
            BPKEEP['#chrom'] = BPKEEP['#chrom'].astype(str)
            BPKEEP.to_csv(self.outpre+'.Keep', sep='\t', index=False)
            self.BPKEEP(BPKEEP)
        else:
            self.log.CW('cannot find the valid files.')
        self.log.CI('finish stating all samples region.')

    def PlotLM(self, _info):
        self._getkeep()
        SampID = _info.sampleid.tolist()
        ecCOL = ['plasmid', 'support_num']  + SampID

        indf = pd.read_csv(self.outpre+'.Keep.matrix', sep='\t')
        indf.rename(columns={'#chrom' : 'plasmid'}, inplace=True)
        indf['plasmid'] = indf['plasmid'].str.upper()

        dPCR = '/data/zhouwei/01Projects/03ecDNA/Nanopore/spikei.info.txt'
        dPCR = pd.read_csv(dPCR, sep='\t')
        dPCR['plasmid'] = dPCR['plasmid'].str.upper()

        ecdf = self.arg.outdir + '/04.EcRegion/All.circle.region.UpMerge_sort'
        ecdf = pd.read_csv(ecdf, sep='\t')
        ecdf.rename(columns={'#chrom' : 'plasmid'}, inplace=True)
        ecdf['plasmid'] = ecdf['plasmid'].str.upper()


        indf = indf.merge(dPCR, on='plasmid', how='right')
        indf.to_csv('./aa.xls', sep='\t', index=False)
        print(indf)
