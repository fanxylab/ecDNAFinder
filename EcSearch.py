#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import collections
import time
import pysam
import pandas as pd
import numpy  as np
from joblib import Parallel, delayed

from EcMagiccube import InterV2, InterVs, InterSm, neighbCigar, neighbBP, neighbBP2

class SearchType():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.genome = self.arg.genomecnv
        self.overmaperrors = self.arg.overmaperrors
        self.dropcigarover = self.arg.dropcigarover
        self.dropneighbdup = self.arg.dropneighbdup
        self.maxhtdistance = self.arg.maxhtdistance
        self.maxneighbtwoends = self.arg.maxneighbtwoends
        self.maxneighboneend = self.arg.maxneighboneend
        self.neighbmergeways = self.arg.neighbmergeways
        self.maxmasksofttwoends = self.arg.maxmasksofttwoends
        self.maxoverlap  = self.arg.maxoverlap
        self.minbplenght = self.arg.minbplenght
        self.minalignlenght = self.arg.minalignlenght
        self.maxbpdistance = self.arg.maxbpdistance
        self.maxmaskallmissmap = self.arg.maxmaskallmissmap
        self.minneighbplen = self.arg.minneighbplen
        self.mingap = self.arg.mingap
        self.maxneiboverlap = self.arg.maxneiboverlap
        self.chrs=[str(i) for i in range(1,23)] + ['MT','X','Y'] \
                    + ['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv',
                        '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                        'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                        'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                        'SunTag-CRISPRi', 'V7-MC-HG-FA']
        self.circle_chrs = ['MT','2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv',
                        '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                        'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                        'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                        'SunTag-CRISPRi', 'V7-MC-HG-FA']

    def _getbeddb(self):
        #inbed = '{0}/{1}/{1}.chimeric.bed'.format(self.arg.Fetch, self.inid)
        inbed = '{0}/{1}/{1}.chimeric.gz'.format(self.arg.Fetch, self.inid) #change 20210121
        inbed = pd.read_csv( inbed, sep='\t', low_memory=False)

        if inbed.empty:
            inbed=pd.DataFrame( columns = inbed.columns.tolist() + ['raw_order', 'fflag'])
        else:
            inbed[['start', 'end']]  = inbed[['start', 'end']].astype(int)
            inbed['#chrom']  = inbed['#chrom'].astype(str)
            inbed['cigarreg'] = inbed.cigarreg.map(eval)
            inbed['fflag']  = 'DROP'
            inbed['raw_order'] = 1

        COLs  = ['#chrom', 'start', 'end',  'SID', 'length', 'forword', 'query_name', 'query_length',
                 'fflag', 'raw_order','query_counts',  'cigarreg', 'mapped_reads']
        inbed = inbed[COLs]
        inbed.sort_values(by=['query_name', 'cigarreg', '#chrom', 'start', 'end' ], ascending=[True]*5, inplace=True)
        inbed['raw_order'] =  inbed.groupby(by=['SID', 'query_name'], sort=False)['raw_order'].apply(np.cumsum)

        cnv_file = '%s/%s/%s.cnv' % (self.arg.CNV, self.inid, self.inid)
        self.cnv_dict = collections.defaultdict(list)
        with open(cnv_file) as f:
            for line in f.readlines()[1:]:
                chrom, start, end, gc, rmsk, counts, bases, counts_md, bases_md, copy, meanCN = line.strip().split('\t')
                try:
                    self.cnv_dict[chrom].append([int(start), int(end), float(copy), int(meanCN)])
                except:
                    self.cnv_dict[chrom].append([int(start), int(end), 0.0, 1])

        #if self.arg.Chrom:
        #    inbed  = inbed[ (inbed['#chrom'].isin(self.chrs) )]
        return inbed

    def cal_gc_in_region(self, _chr, _start, _end):
        seq = self.ref.fetch(_chr, _start, _end)
        length = len(seq)
        length -= seq.count('N') + seq.count('n')
        gc = sum(seq.count(x) for x in 'GCgc')
        return gc / length

    def count_base_in_region(self, _chr, _start, _end):
        min_mapq=0
        min_lenc = 10
        def filter_read(read):
            return not (read.is_duplicate
                        or read.is_secondary
                        or read.is_unmapped
                        or read.is_qcfail
                        or read.mapping_quality < min_mapq)
        try:
            bamfetch = self.samfile.fetch(_chr, _start, _end, multiple_iterators=True)
        except (ValueError, ArithmeticError) :
            print('invalid region %s:%d-%d' % (_chr, _start, _end))
            return 0
        count = 0
        for read in bamfetch:
            if filter_read(read):
                # Only count the bases aligned to the region
                rlen = read.query_alignment_length
                if read.reference_start < _start:
                    rlen -= _start - read.reference_start
                if read.reference_end > _end:
                    rlen -= read.reference_end - _end
                rlen = abs(rlen) # del len lead to negative value
                if rlen >= min_lenc:
                    count += 1
        return count

    def getGC(self, series):
        chrom = series['#chrom']
        start = series['start']
        end = series['end']
        return self.cal_gc_in_region(chrom, start, end)

    def getDepth(self, series):
        chrom = series['#chrom']
        start = series['start']
        end = series['end']
        depth = self.count_base_in_region(chrom, start, end)
        scaled_depth = depth * 5000 / abs(end - start)
        return scaled_depth

    def getCNV(self, series):
        chrom = series['#chrom']
        start = series['start']
        end = series['end']
        if chrom in self.circle_chrs:
            return np.nan
        cnv_list = self.cnv_dict[chrom]
        start_region = [start > x[0] for x in cnv_list]
        end_region = [end < x[1] for x in cnv_list]
        region_idx = [x & y for x, y in zip(start_region, end_region)]
        cnv_region = [x for x,y in zip(cnv_list, region_idx) if y]
        if not cnv_region:
            return 0
        if len(cnv_region) > 1:
            print(chrom, start, end, cnv_region)
        return np.mean([x[2] for x in cnv_region])

    def _getbam(self, _info):
        if 'bamfile' in _info.keys():
            inbam = _info.bamfile
        else:
            inbam = '%s/%s.rmdup.bam'%(self.arg.Bam, _info.sampleid)
            if not os.path.exists(inbam):
                inbam = '%s/%s/%s.rmdup.bam'%(self.arg.Bam, _info.sampleid, _info.sampleid)
        if not os.path.exists(inbam):
            self.log.CW('The bam file of Sample %s cannot be found. Please input right path'%_info.sampleid)
        self.samfile = pysam.AlignmentFile(inbam, 'rb')

    def _getinfo(self, _info):
        self._getbam(_info)
        self.ref = pysam.FastaFile(self.genome)
        self.info = _info
        self.inid = _info.sampleid
        self.outdir= '%s/%s'%(self.arg.Search, self.inid)
        self.arg.outpre= '%s/%s'%(self.outdir, self.inid)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def dropCigarOver(self, _G,  errors=100):
        errors = self.overmaperrors  #100
        _G['fflag'] = neighbCigar(_G[['#chrom', 'start', 'end', 'forword', 'fflag', 'cigarreg']].values,
                                    errors=errors)
        return _G

    def dupliNeighb(self, overm, _l, _n=-1, maxdistance=250, maxsimilar=100, maxreg=True, oriant=True):
        '''
        DUPLIC1 : cover
        DUPLIC2 : overlap
        '''
        maxdistance = self.maxneighbtwoends #250
        maxsimilar  = self.maxneighboneend  #100
        maxreg      = self.neighbmergeways  #True

        _L = overm[_n].copy()
        if (_l['#chrom']  == _L['#chrom']) and (_l['forword'] == _L['forword']):
            D1 = (_L.start <= _l.start + maxsimilar and _L.end >= _l.end - maxsimilar) or \
                    (_l.start <= _L.start + maxsimilar and _l.end >= _L.end - maxsimilar)
            D2 = (np.abs(_L.end   - _l.end) <= maxdistance) and \
                    (np.abs(_L.start - _l.start) <= maxdistance)

            if D1 or D2:
                if   _L['fflag'] in ['DUPMER1','DUPMER2'] :
                    overm.pop(_n)
                else:
                    overm[_n]['fflag'] = 'DUPLIC1' if D1  else 'DUPLIC2'

                if _l['fflag'] not in ['DUPMER1','DUPMER2'] :
                    _l['fflag'] = 'DUPLIC1' if D1  else 'DUPLIC2'
                    overm.append(_l.copy())

                _l['start']  = min([_L['start'], _l['start']])
                _l['end']    = max([_L['end'],   _l['end']]  )
                _l['length']   = _l['end'] - _l['start'] + 1
                _l['cigarreg'] = (min(_L['cigarreg']  + _l['cigarreg']), max(_L['cigarreg']  + _l['cigarreg']))
                _l['fflag'] = 'DUPMER1' if D1  else 'DUPMER2'

                overm.append(_l)
            else:
                overm.append(_l)
        else:
            overm.append(_l)
        return overm

    def mergeNeighb(self, _G):
        if _G.shape[0] <= 1:
            return _G
        else:
            overm = [ _G.iloc[0,:].copy() ]
            for _n, _l in _G.iloc[1:,:].iterrows():
                overm = self.dupliNeighb(overm, _l)

            if  _G.shape[0] > 2:
                overm = self.dupliNeighb(overm[:-1],  overm[-1], 0)

            overm = pd.concat(overm, axis=1).T
            return overm

    def markSingle(self, _G):
        if _G.shape[0] <= 1:
            _G['fflag'] = 'SINGLE'
        return _G

    def markContinuous(self, _G):
        frag_num = _G.shape[0]
        if frag_num < 2:
            return _G
        for i in range(frag_num-1):
            first = _G.loc[_G.index[i]]
            second = _G.loc[_G.index[i+1]]
            if first['cigarreg'][-1] + self.minneighbplen < second['cigarreg'][0]:
                _G['fflag'] = 'DISPERSE'
                return _G
            if first['cigarreg'][-1] > second['cigarreg'][0] + self.maxneiboverlap:
                _G['fflag'] = 'OVERLAP'
                return _G
            if first['#chrom'] != second['#chrom']:
                _G['fflag'] = 'DIFF'
        return _G


    def markBP(self, _G, minneighbplen=500, mingap=100, maxneiboverlap=500):
        minneighbplen = self.minneighbplen
        mingap = self.mingap
        maxneiboverlap = self.maxneiboverlap
        _G['fflag'] = neighbBP2(_G[['#chrom', 'start', 'end', 'forword', 'fflag', 'cigarreg']].values,
                                    minneighbplen=minneighbplen, mingap=mingap, maxneiboverlap=maxneiboverlap)
        return _G

    def mergeBP(self, _G):
        _G = _G.copy()
        _B = _G[(_G.fflag.str.contains('EcBP'))]
        _H = _B.iloc[0,:]
        _T = _B.iloc[-1,:]
        _L = _H.copy()

        _G.loc[_H.name, 'fflag'] += ';HEAD'
        _G.loc[_T.name, 'fflag'] += ';TAIL'

        _L['start']  = min([_H['start'], _T['start']])
        _L['end']    = max([_H['end'],   _T['end']])
        _L['length'] = _L['end'] - _L['start'] + 1
        _L['fflag']  += ';HTBREAKP'
        _L['cigarreg'] = (min(_H['cigarreg']  + _T['cigarreg']), max(_H['cigarreg']  + _T['cigarreg']))
        _G = _G.append(_L)
        return _G

    def typeCat(self, indf, dropcigarover=True, dropneighbdup=True, minalignlenght=100):
        dropcigarover = self.dropcigarover #True
        dropneighbdup = self.dropneighbdup #True
        GRPBY = ['SID', 'query_name']

        # drop by cnv
        indf['cnv'] = indf.apply(self.getCNV, axis=1)
        #indf['gc'] = indf.apply(self.getGC, axis=1)
        #indf['scaled_depth'] = indf.apply(self.getDepth, axis=1) # too much time
        ## scale cnv
        #info_file = self.arg.outpre+'.info'
        #indf.to_csv(info_file, sep='\t', index=False)
        #cnv_file = '%s/%s/%s.cnv' % (self.arg.CNV, self.inid, self.inid)
        #data_file = info_file.replace('.info', '.data')
        #cnv_cmd = '/datd/enzedeng/software/R-4.1.1/bin/Rscript %s/scaleCNV.R %s %s %s' % (
        #    self.arg.scriptdir, cnv_file, info_file, data_file)
        #os.system(cnv_cmd)

        #indf = pd.read_csv(data_file, sep='\t', dtype={'#chrom':str})
        #indf[['start', 'end']] = indf[['start', 'end']].astype(int)
        #indf['cigarreg'] = indf['cigarreg'].map(eval)
        #indf['copy'] = indf['copy'].astype(float)

        indf.loc[indf['cnv'] < 2,'fflag'] = 'LOWCNV'
        #indf.loc[indf['copy'] < 2,'fflag'] = 'LOWCNV'
        LOWC = indf[(indf.fflag=='LOWCNV')]
        #indf = indf[(indf.fflag!='LOWCNV')]

        self.log.CI('start droping overlap of mapping region: ' + self.inid)
        # drop min align lenght
        indf.loc[ (np.abs(indf.cigarreg.str[1] - indf.cigarreg.str[0]) < self.minalignlenght - 1), 'fflag'] ='LOWALIGN'
        LOWA = indf[(indf.fflag=='LOWALIGN')]
        indf = indf[(indf.fflag!='LOWALIGN')]

        # dropcigarover
        self.log.CI('start droping overlap of cigar: ' + self.inid)
        if dropcigarover:
            indf =  Parallel( n_jobs= -1, backend='threading')( delayed( self.dropCigarOver )(_g)
                               for _, _g in indf.groupby(by=GRPBY, sort=False))
            indf = pd.concat(indf, axis=0, sort=False)
        OVER = indf[(indf.fflag=='OVER')]
        indf = indf[(indf.fflag!='OVER')]

        # mergeNeighb
        self.log.CI('start merging neighbour duplcations of mapping region: ' + self.inid)
        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.mergeNeighb )(_g)
                    for _, _g in indf.groupby(by=GRPBY, sort=False))
        indf = pd.concat(indf, axis=0, sort=False)
        DUPL = indf[ (indf.fflag.str.contains('DUPLIC', regex=False))]
        indf = indf[~(indf.fflag.str.contains('DUPLIC', regex=False))]

        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.markSingle )(_g)
                    for _, _g in indf.groupby(by=GRPBY, sort=False))
        indf = pd.concat(indf, axis=0, sort=False)
        SING = indf[(indf.fflag=='SINGLE')]
        indf = indf[(indf.fflag!='SINGLE')]

        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.markContinuous )(_g)
                    for _, _g in indf.groupby(by=GRPBY, sort=False))
        indf = pd.concat(indf, axis=0, sort=False)
        DIOV = indf[(indf.fflag=='DISPERSE') | (indf.fflag=='OVERLAP')]
        indf = indf[(indf.fflag!='DISPERSE') & (indf.fflag!='OVERLAP')]

        MARK = pd.concat( [LOWC, LOWA, SING, OVER, DUPL, DIOV, indf], axis=0, sort=False)
        del(LOWC, LOWA, OVER, DUPL, SING, DIOV)
        DIFF = indf[indf.fflag == 'DIFF']
        SAME = indf[indf.fflag != 'DIFF']
        DIFF.to_csv(self.arg.outpre+'.DIFF', sep='\t', index=False)
        SAME.to_csv(self.arg.outpre+'.SAME', sep='\t', index=False)
        MARK.to_csv(self.arg.outpre+'.Mark', sep='\t', index=False)
        indf.to_csv(self.arg.outpre+'.Keep', sep='\t', index=False)
        del(MARK, indf)
        return

        # markEcBP
        self.log.CI('start marking and merging head-to-tail mapping region: ' + self.inid)
        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.markBP )(_g)
                   for _, _g in indf.groupby(by=GRPBY, sort=False))
        indf = pd.concat(indf, axis=0, sort=False)

        # TODO one read can be made of three fragments
        indf['bps'] = indf.groupby(by=GRPBY, sort=False)['fflag'].transform(lambda x:len([i for i in x if 'EcBP' in i ]))
        LINE = indf[ ((indf.bps < 2) | (indf.fflag =='INTER')) ]
        indf = indf[ ((indf.bps >=2) & (indf.fflag !='INTER')) ]
        LINE.drop('bps', axis=1, inplace=True)
        indf.drop('bps', axis=1, inplace=True)

        # mergeBP
        self.log.CI('start merging head-to-tail mapping region: ' + self.inid)
        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.mergeBP )(_g)
                    for _, _g in indf.groupby(by=GRPBY, sort=False))
        indf = pd.concat(indf, axis=0, sort=False)

        # concat
        MARK = pd.concat( [LOWC, LOWA, SING, OVER, DUPL, LINE, indf], axis=0, sort=False)
        del(LOWC, LOWA, SING, OVER, DUPL, LINE, )

        # headtailregion
        self.log.CI('start adding heat/tail site to a new column: ' + self.inid)
        KEEP = indf.merge(indf.groupby(by=GRPBY, sort=False)\
                            .apply(lambda x:
                                x.loc[(x.fflag.str.contains(';HEAD|;TAIL')), ['start','end']].values.tolist())\
                            .to_frame(name='HTSites').reset_index(), on=GRPBY)
        KEEP = KEEP[ ~(KEEP.fflag.str.contains('HEAD|TAIL', regex=True))]

        MARK.to_csv(self.arg.outpre+'.Mark', sep='\t', index=False)
        KEEP.to_csv(self.arg.outpre+'.Keep', sep='\t', index=False)
        del(MARK, KEEP, indf)

    def TypeBase(self, _info):
        self._getinfo(_info)
        self.log.CI('start searching breakpoin region: ' + self.inid)
        inbed = self._getbeddb()

        if not inbed.empty:
            self.typeCat( inbed )
        else:
            self.log.CW('cannot find any circle region singal: ' + self.inid)
        self.log.CI('finish searching breakpoin region: ' + self.inid)
