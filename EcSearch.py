#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy  as np
from joblib import Parallel, delayed

from EcMagiccube import InterV2, InterVs, InterSm, neighbCigar, neighbBP, neighbBP2

class SearchType():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
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

        #if self.arg.Chrom:
        #    inbed  = inbed[ (inbed['#chrom'].isin(self.chrs) )]
        return inbed

    def _getinfo(self, _info):
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

        # markEcBP
        self.log.CI('start marking and merging head-to-tail mapping region: ' + self.inid)
        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.markBP )(_g)
                   for _, _g in indf.groupby(by=GRPBY, sort=False))
        indf = pd.concat(indf, axis=0, sort=False)

        indf['bps'] = indf.groupby(by=GRPBY, sort=False)['fflag'].transform(lambda x:len([i for i in x if 'EcBP' in i ]))
        LINE = indf[ ((indf.bps !=2) | (indf.fflag =='INTER')) ]
        indf = indf[ ((indf.bps ==2) & (indf.fflag !='INTER')) ]
        LINE.drop('bps', axis=1, inplace=True)
        indf.drop('bps', axis=1, inplace=True)

        # mergeBP
        self.log.CI('start merging head-to-tail mapping region: ' + self.inid)
        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.mergeBP )(_g)
                    for _, _g in indf.groupby(by=GRPBY, sort=False))
        indf = pd.concat(indf, axis=0, sort=False)

        # concat
        MARK = pd.concat( [LOWA, OVER, DUPL, LINE, indf], axis=0, sort=False)
        del(LOWA, OVER, DUPL, LINE, )

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


class SearchType1():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
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
        self.chrs=[str(i) for i in range(1,23)] + ['MT','X','Y'] \
                    + ['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv', 
                        '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                        'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                        'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                        'SunTag-CRISPRi', 'V7-MC-HG-FA']

    def _getinfo(self, _info):
        self.info = _info
        self.inid = _info.sampleid
        self.outdir= '%s/%s'%(self.arg.Search, self.inid)
        #self.arg.outpre= '%s/%s%s'%(self.outdir, self.inid, self.Chrom)
        self.arg.outpre= '%s/%s'%(self.outdir, self.inid)
        os.makedirs(self.outdir, exist_ok=True)
        return self

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
                 'fflag', 'raw_order','query_counts', 'cigarstring',  'cigarreg', 'mapped_reads']
        inbed = inbed[COLs]
        inbed.sort_values(by=['query_name', 'cigarreg', '#chrom', 'start', 'end' ], ascending=[True]*5, inplace=True)
        inbed['raw_order'] =  inbed.groupby(by=['SID', 'query_name'], sort=False)['raw_order'].apply(np.cumsum)

        #if self.arg.Chrom:
        #    inbed  = inbed[ (inbed['#chrom'].isin(self.chrs) )]
        return inbed

    def dropCigarOver(self, _G,  errors=100):
        errors = self.overmaperrors  #100
        if _G.shape[0] <2:
            return _G
        else:
            _G = _G.reset_index(drop=True).copy()
            for _n, _l in _G.iloc[:-1,:].iterrows():
                for _m, _k in _G.loc[_n+1:,:].iterrows():
                    _s1 = _l.cigarreg[0]
                    _e1 = _l.cigarreg[1]
                    _f1 = _l.forword
                    _s2 = _k.cigarreg[0]
                    _e2 = _k.cigarreg[1]
                    _f2 = _k.forword
                    if   (_s1 <= _s2 + errors) and (errors + _e1 >= _e2) and (_f1 == _f2):
                        _G.loc[_m, 'fflag'] ='OVER'
                    elif (_s2 <= _s1 + errors) and (errors + _e2 >= _e1) and (_f1 == _f2):
                        _G.loc[_n, 'fflag'] ='OVER'
            return _G

    def maxBedDistance(self, _G):
        if _G.shape[0] <2:
            return _G
        elif _G.shape[0] ==2: #reduce running time
            if _G['#chrom'].unique().size == _G['forword'].unique().size == 1:
                _G['fflag'] = 'HTDIST'
            return _G
        else:
            if _G['#chrom'].unique().size == _G.shape[0]: #reduce running time
                return _G
            else:
                _G = _G.reset_index(drop=True).copy()
                k=[]
                for _n, _l in _G.iterrows() :
                    for _m, _j in _G.loc[_n+1:].iterrows():
                        if (_l['#chrom'] == _j['#chrom']) and (_l['forword'] == _j['forword']):
                            k.append( [_n ,_m, _m - _n, _G.loc[_n:_m,'length'].sum() ])
                if k:
                    k = pd.DataFrame(k, columns=['s','e','d','l'])\
                            .sort_values(by=['l','d'],ascending=[False, False])\
                            .iloc[0,:]
                    _G.loc[k.s : k.e, 'fflag'] = 'HTDIST'
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

                if _L['cigarstring'] not in _l['cigarstring']:
                    _l['cigarstring'] = '%s;%s'%(_L['cigarstring'], _l['cigarstring'])

                overm.append(_l)
            else:
                overm.append(_l)
        else:
            overm.append(_l)
        return overm

    def mergeNeighb(self, _G): #bug ,need to change
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

    def mergeHeadTail(self, _G, maxhtdistance=10000000, maxbpdistance=300 ):
        maxhtdistance = self.maxhtdistance  #10000000 deprecated
        maxbpdistance = self.maxbpdistance  #300
        _G = _G.reset_index(drop=True)
        _H = _G.iloc[0,:]
        _T = _G.iloc[-1,:]
        _L = _G.iloc[0,:].copy()

        #if _L > maxhtdistance:
        #    print('warning: the ecDNA breakpoint lenght is large than %s:\n%s'%(_L, _G))

        _G.loc[_H.name, 'fflag'] += ';HEAD'
        _G.loc[_T.name, 'fflag'] += ';TAIL'

        _L['start']  = min([_H['start'], _T['start']])
        _L['end']    = max([_H['end'],   _T['end']])
        _L['length'] = _L['end'] - _L['start'] + 1
        _L['fflag']  += ';HTBREAKP'
        _L['cigarreg']   = (min(_H['cigarreg']  + _T['cigarreg']), max(_H['cigarreg']  + _T['cigarreg']))
        _L['cigarstring'] = '%s;%s'%(_H['cigarstring'], _T['cigarstring'] )
        _G = pd.concat([_L.to_frame().T, _G], axis=0, sort=False)
        return _G

    def markKeep(self, _G, maxoverlap=1000, maxmiste=0.15, maxmisal=0.35):
        maxoverlap = self.maxoverlap    #1000
        maxmiste = self.maxmasksofttwoends #0.15
        maxmisal = self.maxmaskallmissmap  #0.35

        if   _G.shape[0] <2:
            return _G
        elif _G.shape[0]>=2:
            BreakF = _G.iloc[ 0,:]
            BreakL = _G.iloc[-1,:]
            C  = BreakF['#chrom'] == BreakL['#chrom']
            F  = BreakF.forword == BreakL.forword
            if BreakF.cigarreg[0] > BreakF.query_length*maxmiste:
                _G['fflag'] +=';FRONTMISS'
            if BreakL.cigarreg[1] < BreakF.query_length*(1-maxmiste):
                _G['fflag'] +=';ENDMISS'
            if InterSm(InterVs(_G.cigarreg.map(list).tolist()))/BreakF.query_length < 1- maxmisal: 
                #if DUPMER1, there will get a high mapping ration
                _G['fflag'] +=';MAPMISS'
            S1 = BreakF.start >= BreakL.start
            E1 = BreakF.end   >= BreakL.end
            X1 = BreakF.start >= BreakL.end - maxoverlap  ###need to change
            O1 = BreakF.start <= BreakL.end

            S2 = BreakF.start <= BreakL.start
            E2 = BreakF.end   <= BreakL.end
            X2 = BreakL.start >= BreakF.end - maxoverlap  ###need to change
            O2 = BreakL.start <= BreakF.end

            if (C and F):
                if   (S1 and E1 and X1 and BreakF.forword=='+' ):
                    _G['fflag'] += ';EcDNA;CiR' if O1 else ';EcDNA'
                elif (S2 and E2 and X2 and BreakF.forword=='-' ):
                    _G['fflag'] += ';EcDNA;CiR' if O2 else ';EcDNA'
                else:
                    _G['fflag'] +=';TRANS'
            return _G

    def typeCat(self, indf, dropcigarover=True, dropneighbdup=True, minalignlenght=100):
        dropcigarover = self.dropcigarover #True
        dropneighbdup = self.dropneighbdup #True
        GRPBY = ['SID', 'query_name']

        self.log.CI('start droping overlap of mapping region: ' + self.inid)
        # drop min align lenght
        indf.loc[ (np.abs(indf.cigarreg.str[1] - indf.cigarreg.str[0]) < self.minalignlenght - 1), 'fflag'] ='LOWALIGN'
        LOWA = indf[(indf.fflag=='LOWALIGN')]
        indf = indf[(indf.fflag!='LOWALIGN')]

        # dropcigarover
        if dropcigarover:
            indf =  Parallel( n_jobs= -1, backend='threading')( delayed( self.dropCigarOver )(_g)
                               for _, _g in indf.groupby(by=GRPBY, sort=False))
            indf = pd.concat(indf, axis=0, sort=False)
        OVER = indf[(indf.fflag=='OVER')]
        indf = indf[(indf.fflag!='OVER')]

        # maxbeddistance
        self.log.CI('start computing maximal distance of mapping region: ' + self.inid)
        indf =  Parallel( n_jobs= -1, backend='threading')( delayed( self.maxBedDistance )(_g)
                            for _, _g in indf.groupby(by=GRPBY, sort=False))
        indf = pd.concat(indf, axis=0, sort=False)
        DIST = indf[(indf.fflag!='HTDIST')]
        indf = indf[(indf.fflag=='HTDIST')]

        # mergeNeighb
        self.log.CI('start merging neighbour duplcations of mapping region: ' + self.inid)
        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.mergeNeighb )(_g)
                    for _, _g in indf.groupby(by=GRPBY, sort=False))
        indf = pd.concat(indf, axis=0, sort=False)
        DUPL = indf[ (indf.fflag.str.contains('DUPLIC', regex=False))]
        indf = indf[~(indf.fflag.str.contains('DUPLIC', regex=False))]

        # markEcDNA
        self.log.CI('start marking and merging head-to-tail mapping region: ' + self.inid)
        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.markKeep )(_g)
                    for _, _g in indf.groupby(by=GRPBY, sort=False))
        indf = pd.concat(indf, axis=0, sort=False)
        LINE = indf[~((indf.fflag.str.contains('EcDNA')) & ~(indf.fflag.str.contains('MISS')))]
        indf = indf[ ((indf.fflag.str.contains('EcDNA')) & ~(indf.fflag.str.contains('MISS')))]

        # mergeHeadTail
        self.log.CI('start merging head-to-tail mapping region: ' + self.inid)
        indf = Parallel( n_jobs= -1, backend='threading')( delayed( self.mergeHeadTail )(_g)
                    for _, _g in indf.groupby(by=GRPBY, sort=False))
        indf = pd.concat(indf, axis=0, sort=False)

        # concat
        MARK = pd.concat( [LOWA, OVER, DIST, DUPL, LINE, indf], axis=0, sort=False)
        del(LOWA, OVER, DIST, DUPL, LINE, )

        # headtailregion
        self.log.CI('start adding heat/tail site to a new column: ' + self.inid)
        KEEP = indf.merge(indf.groupby(by=GRPBY, sort=False)\
                            .apply(lambda x: 
                                x.loc[(x.fflag.str.contains(';HEAD|;TAIL')), ['start','end']].values.tolist())\
                            .to_frame(name='HTSites').reset_index(), on=GRPBY)
        #KEEP.loc[~KEEP.fflag.str.contains('HTBREAKP'), 'HTSites'] = ''
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
