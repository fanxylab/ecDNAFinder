#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from Bio import SeqIO
from Bio.Seq import Seq
from joblib import Parallel, delayed
import pandas as pd
import gzip
import re

from .EcUtilities import Utilities

class SeqEngine():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.outdir= self.arg.Region
        self.arg.outpre= '%s/%s'%(self.arg.Region,self.arg.regionpre)

    def _getUpmerge(self):
        if self.arg.linkfile:
            self.upmerge = self.arg.linkfile
            #self.upmerge = '%s/%s.UpMerge_sort.nanopore.20201117.xls'%(self.arg.Region,self.arg.regionpre)
        else:
            self.upmerge = '%s/%s.UpFilter'%(self.arg.Region,self.arg.regionpre)
            self.upmerge = '%s/%s.UpFilterTRF'%(self.arg.Region,self.arg.regionpre)
            self.Links = '%s/%s.Links'%(self.arg.Region,self.arg.regionpre)


        self.upmerge= pd.read_csv(self.upmerge, sep='\t', low_memory=False)
        self.upmerge.drop(['gene_name','gene_biotype'], axis=1, inplace=True)
        self.upmerge['#chrom'] = self.upmerge['#chrom'].astype(str)
        self.upmerge['start']  = self.upmerge['start'].astype(int)
        self.upmerge['end']    = self.upmerge['end'].astype(int)

        CoLink = ['SID', 'query_name', 'forword', 'cigarstring', 'cigarreg', 'LINKS', '#chrom', 'start_n', 'end_n']
        self.Links = pd.read_csv(self.Links, sep='\t', low_memory=False)
        self.Links = self.Links.loc[(self.Links.LINKS.isin(self.upmerge.LINKS)), CoLink]
        self.Links = self.Links.sort_values(by='forword')\
                         .groupby('LINKS')\
                         .apply(lambda x:x[(x.query_name ==x.query_name.values[0])])\
                         .reset_index(drop=True)
        self.Links.cigarreg = self.Links.cigarreg.map(eval)

    def refFecth(self, _g, _S):
        _L = int(min(self.arg.lengthbpseq, len(_S)))
        if _g.forword =='-':
            _S = _S.reverse_complement()
        _S = str(_S.seq)
        _g['F_Ref'] = _S[:_L]
        _g['R_Ref'] = _S[-_L:]
        return _g

    def transcigar(self, _st, forword):
        CICAR = {'M':0, 'I':1, 'D':2, 'N':3, 'S':4, 'H':5, 'P':6, '=':7, 'X':8}
        _tuple = zip(map(lambda x: CICAR[x], re.split('[0-9]+', _st)[1:]),
                    map(int, re.split('[A-Z=]' , _st)[:-1]))
        _tuple = list(_tuple)

        if forword=='-':
            _tuple = _tuple[::-1]

        cigarpos = [(0, _tuple[0][1])]
        for i in _tuple[1:]:
            cigarpos.append( (cigarpos[-1][1], cigarpos[-1][1]+i[1]) )
        
        for  n,i in enumerate(_tuple):
            if (i[0] ==0 ):
                return cigarpos[n]

    def getQuery(self, _l, _S):
        #if _l.forword =='-':
        #    _S = _S.reverse_complement()
        _S = str(_S.seq)
        Cir= _l.cigarstring.split(';')
        if len(Cir) ==1:
            _S = _S[_l.cigarreg[0]:_l.cigarreg[1]]
            _L  = int(min(self.arg.lengthbpseq, len(_S)))
            _l['F_Que'] = _S[:_L]
            _l['R_Que'] = _S[-_L:]
        else:
            _C = [ self.transcigar(_n, _l.forword ) for _n in Cir ]

            _F = sorted(_C, key=lambda x:x[0])
            _Fs= _F[0][0]
            _Fe= _F[0][1]
            _Fs= int(max(_Fe-self.arg.lengthbpseq, _Fs))

            _R = sorted(_C, key=lambda x:x[1])
            _Rs= _F[-1][0]
            _Re= _F[-1][1]
            _Re= int(min(_Rs+self.arg.lengthbpseq, _Re))
            if _l.forword == '+':
                _l['R_Que'] = _S[_Fs:_Fe]
                _l['F_Que'] = _S[_Rs:_Re]
            else:
                _l['F_Que'] = _S[-_Fe:-_Fs]
                _l['R_Que'] = _S[-_Re:-_Rs]
        return _l

    def queFetch(self, _g):
        SID   = _g.SID.values[0]
        fasta = '{0}/{1}/{1}.chimeric.fasta.gz'.format(self.arg.Fetch, SID ) 
        #the fasta may save a comreverse  seq, need change to use the raw fastq file
        #fasta = Utilities('','').readfasta(fasta)
        with gzip.open(fasta, "rt") as handle:
            fasta = SeqIO.to_dict(SeqIO.parse(handle, "fasta"))
        _g = Parallel( n_jobs=-1, backend='loky')( delayed(self.getQuery)( _l, fasta[_l.query_name])
                            for _n, _l in _g.iterrows())
        del fasta
        _g = pd.concat(_g, axis=1, sort=False).T
        return _g

    def GetBPSeq(self, _info):
        self._getUpmerge()

        QueSeq = Parallel( n_jobs= 3, backend='loky')( delayed( self.queFetch )(_g) for _n, _g in self.Links.groupby('SID') )
        QueSeq = pd.concat(QueSeq, axis=0, sort=False)
        QueSeq.rename(columns={'start_n': 'start', 'end_n': 'end'}, inplace=True)
        QueSeq.to_csv('%s.UpFilter.BP%s.log'%(self.arg.outpre, self.arg.lengthbpseq), sep='\t', index=False)
        QueSeq = QueSeq[['#chrom', 'start', 'end', 'LINKS', 'F_Que', 'R_Que']]

        Ref = SeqIO.to_dict(SeqIO.parse(self.arg.genome, "fasta"))
        BPSeq = Parallel( n_jobs=-1, backend='loky')( delayed( self.refFecth )(_l, Ref[_l['#chrom']][_l.start:_l.end])
                            for _n, _l in self.upmerge.iterrows() )
        BPSeq = pd.concat(BPSeq, axis=1, sort=False).T
        del Ref

        BPSeq = BPSeq.merge(QueSeq, on =['#chrom', 'start', 'end', 'LINKS'], how='left')
        BPSeq.to_csv('%s.UpFilter.BP%s.xls'%(self.arg.outpre, self.arg.lengthbpseq), sep='\t', index=False)

