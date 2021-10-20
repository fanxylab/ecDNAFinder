#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import numpy  as np
import pybedtools as bt
from joblib import Parallel, delayed
import os
import gzip
from EcMagiccube import MaxBetween

class Utilities():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log

    def emptyfile(_func):
        def wrapper(self, *args, **kargs):
            file = args[0]
            if os.path.exists(file):
                sz = os.path.getsize(file)
                if not sz:
                    print(file, " is empty!")
                else:
                    _func(self, *args, **kargs)
            else:
                print(file, " is not exists!")
        return wrapper

    def readgtf(self, gtffile):
        def splitinfo(_l):
            _l = _l.strip(';').split('; ')
            _k = ['']*3
            for _ll in _l:
                if  _ll.split(' ')[0]=='gene_id':
                    _k[0] = _ll.split(' ')[1].strip('"')
                if  _ll.split(' ')[0]=='gene_name':
                    _k[1] = _ll.split(' ')[1].strip('"')
                if  _ll.split(' ')[0]=='gene_biotype':
                    _k[2] = _ll.split(' ')[1].strip('"')
            return _k
        with open(gtffile, 'r') as  f:
            gtf = [ i.strip().split('\t')[:-1] + splitinfo(i.strip().split('\t')[-1]) 
                    for i in f.readlines() 
                    if (not i.startswith('#')) and (i.split('\t')[2]=='gene') ]
        gtf = pd.DataFrame(gtf, columns=['#chrom', 'db', 'gtype', 'start', 'end', 'U1', 'forword',
                                            'U2', 'gene_id', 'gene_name', 'gene_biotype'])
        gtf['length'] = gtf.end.astype(int) - gtf.start.astype(int) + 1
        gtf = gtf[['#chrom', 'start', 'end', 'gtype', 'length', 'forword', 'gene_name', 'gene_id', 'gene_biotype']]
        gtf.to_csv('./Homo_sapiens.GRCh38.100.gtf.gene.bed', sep='\t',index=False)

    def readfasta(self, Ifa):
        fadict ={}
        if Ifa.endswith('.gz'):
            O = gzip.open(Ifa, 'rt')
        else:
            O = open(Ifa, 'r')
        with O as f:
            _k = ''
            for i in f.readlines():
                if i.startswith('>'):
                    fadict[ i.strip('>|\n')] = ''
                    _k = i.strip('>|\n')
                else:
                    fadict[ _k ] += i.strip()
        return fadict

    def savefa(self, fadict, out):
        f=gzip.open(out, 'wb')
        for k,v in fadict.items():  
            f.write(('>%s\n%s\n'%(k,v)).encode())
        f.close()

    def annogene(self, _inbed, minover=30, 
                    biotype=['miRNA','lncRNA', 'protein_coding', 'IG_C_gene', 'IG_D_gene', 
                            'IG_J_gene', 'IG_V_gene', 'TR_C_gene', 'TR_D_gene', 'TR_J_gene']
                            #'TR_V_gene', 'rRNA', 'scaRNA', 'scRNA', 'snoRNA', 'snRNA', 'sRNA'] ):
                ):
        self.GTFBED = self.arg.gtfbed

        GTFBED = pd.read_csv(self.GTFBED, sep='\t', header=0)
        GTFBED = GTFBED[(GTFBED.gene_biotype.isin(biotype))]
        GTFBED['#chrom'] = GTFBED['#chrom'].astype(str)
        _inbed['#chrom'] = _inbed['#chrom'].astype(str)

        COL1 = ['#chrom', 'start', 'end', 'length']
        COL2 = ['#chrom_g', 'start_g', 'end_g', 'gtype', 'length_g', 'forword_g',  'gene_name', 'gene_id', 'gene_biotype']
        LINE = ['.', -1, -1, '.', '.', '.', '.', '.', '.']

        annobed = bt.BedTool.from_dataframe(GTFBED)
        inbed   = bt.BedTool.from_dataframe(_inbed)
        inbed   = inbed.intersect(annobed, s=False, S=False, loj=True)\
                        .to_dataframe(disable_auto_names=True,  header=None, names=COL1+COL2)

        # reduce running time
        inbed['#chrom'] = inbed['#chrom'].astype(str)
        inbed1 = inbed[(inbed.start_g ==-1)]
        inbed2 = inbed[(inbed.start_g !=-1)].copy()
        inbed2['len_m'] = inbed2[['end', 'end_g']].min(1) - inbed2[['start', 'start_g']].max(1) + 1
        inbed2.loc[(inbed2.len_m < minover), COL2] = LINE
        inbed2 = inbed2[COL1 + COL2]

        inbed = pd.concat([inbed1, inbed2], axis=0, sort=False)
        del(inbed1, inbed2)
        #inbed.to_csv(outpre+'.BedAnnotate.gz', sep='\t', index=False, compression='gzip')

        inbed = pd.merge(inbed.groupby(by=COL1)['gene_name'].apply(lambda x:x.astype(str).str.cat(sep=';')).reset_index(),
                         inbed.groupby(by=COL1)['gene_biotype'].apply(lambda x:x.astype(str).str.cat(sep=';')).reset_index(),
                         on=COL1, how='outer')
        return inbed

    def _mapneigtb(self, _indf, sortC = ['#chrom', 'start', 'end', 'forword'], maxdistance=500, maxreg=True, oriant=False):
        indf   = _indf.sort_values(by=sortC, ascending=[True]*len(sortC))
        cols   = ['#chrom', 'start', 'end',  'forword']
        indf   = indf[cols].to_numpy().tolist()
        mapset = [ indf[0] ]
        for _l in indf:
            _m = -1
            _L = mapset[_m]

            R = str(_l[0]) == str(_L[0])
            F = str(_l[3]) == str(_L[3])
            S = (_L[1] - maxdistance) <= _l[1] <= (_L[1] + maxdistance)
            E = (_L[2] - maxdistance) <= _l[2] <= (_L[2] + maxdistance)

            if (R and S and E) and ( not(oriant and F) ):
                if maxreg:
                    _L[1] = min([_L[1], _l[1]])
                    _L[2] = max([_L[2], _l[2]])
                else:
                    _L[1] = max([_L[1], _l[1]])
                    _L[2] = min([_L[2], _l[2]])
                mapset[_m] = _L
            else:
                mapset.append( _l )
        mapset = pd.DataFrame(mapset, columns=cols)
        return mapset

    def linkannot(self, _BPs, _genbed, minover=30):
        _chr  = _BPs[0]
        _tbed = _genbed[ (_genbed[:,0] == _chr) ]
        _tboo = np.zeros((_tbed.shape[0]),dtype=bool)
        for _start, _end in _BPs[1]:
            set1 = ((_tbed[:,1]>= _start) & (_tbed[:,2]<= _end))
            set2 = ((_tbed[:,1]< _start)  & (_tbed[:,2]>= _start+minover))
            set3 = ((_tbed[:,1]<= _end-minover) & (_tbed[:,2]> _end))
            _tboo += (set1 | set2 | set3)
        _tbed[_tboo,-2:]
        return [';'.join(_tbed[_tboo,-2]), ';'.join(_tbed[_tboo,-1])]

    def BTannot(self,  _BP, _tbed, mino=30, trfd=300):
        _tbed = _tbed[(_tbed[:,0]==_BP[0])]
        _tboo = []
        for i in [1,2]:
            (_s, _e) =  (_BP[i], _BP[i]+trfd)  if (i==1) else (_BP[i]-trfd, _BP[i])
            set1 = ((_tbed[:,1]>= _s) & (_tbed[:,2]<=_e)).any()
            set2 = ((_tbed[:,1]<  _s) & (_tbed[:,2]>=_s + mino)).any()
            set3 = ((_tbed[:,1]<= _e - mino) & (_tbed[:,2] > _e)).any()
            if (set1 or set2 or set3):
                _tboo.append( 'Htrf' if i==1 else 'Ttrf')
        return ';'.join(_tboo)

    def mapneigtb(self, _indf, maxdistance=500, maxreg=True, oriant=False):
        if oriant:
            sortA = ['#chrom', 'forword', 'start', 'end']
            sortD = ['#chrom', 'forword', 'end', 'start']
        else:
            sortA = ['#chrom', 'start', 'end', 'forword']
            sortD = ['#chrom', 'end', 'start', 'forword']

        indf  = _indf.copy()
        mline = 0
        while mline != indf.shape[0]:
            mline = indf.shape[0]
            indf = self._mapneigtb( indf, sortD)
            print(indf.shape)
            indf = self._mapneigtb( indf, sortA)
            print(indf.shape)

        return indf

    def mapanytwo1(self, indf, maxdistance=500, maxreg=True, maxline=3000000, oriant=False):
        def _splitmap(_inmap):
            inmap  = _inmap.copy()
            for _n, _l in inmap.iterrows():
                S = inmap.start_n.between(_l.start - maxdistance, _l.start + maxdistance, inclusive=True)
                E = inmap.end_n.between(  _l.end   - maxdistance, _l.end   + maxdistance, inclusive=True)
                inmap.loc[(S & E),'start_n'] = inmap[(S & E)]['start'].min()
                inmap.loc[(S & E),  'end_n'] = inmap[(S & E)]['end'  ].max()
            return inmap

        sortN = ['#chrom', 'start', 'end', 'forword']
        mapsN = ['#chrom', 'start', 'end', 'forword', 'start_n', 'end_n']
        grpby = ['#chrom', 'forword'] if oriant else ['#chrom']

        indf  =  indf.copy().sort_values(by=sortN)
        indf[['start_n', 'end_n']] = indf[['start', 'end']]

        if indf.shape[0] > maxline:
            inmap = indf[mapsN].drop_duplicates(keep='first')
            inmap = Parallel( n_jobs=-1, backend='loky')(delayed(_splitmap )(_g) 
                            for _, _g in inmap.groupby(by=grpby, sort=False))
            inmap = pd.concat(inmap, axis=0)
            indf = indf.merge(inmap, on=sortN, how='left')
        else:
            indf = Parallel( n_jobs=-1, backend='loky')(delayed(_splitmap )(_g) 
                            for _, _g in indf.groupby(by=grpby, sort=False))
            indf = pd.concat(indf, axis=0)

        indf[['start_n', 'end_n']] = indf[['start_n', 'end_n']].astype(int)
        indf[['length_n']] = indf['end_n'] -  indf['start_n'] + 1
        return indf

    def mapanytwo(self, indf, maxdistance=500, maxreg=True, oriant=False):
        sortN = ['#chrom', 'start', 'end', 'forword']
        mapsN = ['#chrom', 'start', 'end', 'forword', 'start_n', 'end_n']
        grpby = ['#chrom', 'forword'] if oriant else ['#chrom']

        indf  =  indf.copy().sort_values(by=sortN)

        inmap = indf[sortN].copy().drop_duplicates(keep='first')
        inmap[['start_n', 'end_n']] = inmap[['start', 'end']]
        inmap = Parallel( n_jobs=-1, backend='loky')(delayed(MaxBetween)(_g.values, maxdistance) 
                        for _, _g in inmap[mapsN].groupby(by=grpby, sort=False))
        inmap = pd.DataFrame(np.vstack(inmap), columns=mapsN)
        indf  = indf.merge(inmap, on=sortN, how='left')
        del (inmap)

        indf[['start_n', 'end_n']] = indf[['start_n', 'end_n']].astype(int)
        indf[['length_n']] = indf['end_n'] -  indf['start_n'] + 1
        return indf

    @emptyfile
    def bambedflter(self, inbam, output):
        self.checkbed = self.arg.checkbed
        self.samtools = self.arg.samtools
        self.bedtools = self.arg.bedtools

        cmd = '''
        {samtools}/samtools view -b -F 256 -F 272 {inbam} | \
        {bedtools}/bedtools intersect -a {checkbed} -b stdin  -wa -wb -loj -bed > {output} 
        '''.format(samtools=self.samtools, 
                    bedtools=self.bedtools, 
                    checkbed=self.checkbed, 
                    inbam=inbam,
                    output=output).strip()
        print(cmd)
        os.system( 'echo " %s" | qsub -V -cwd -pe smp 15 -l h_vmem=60G' %cmd)

    def bedintersect(self, beddfa, beddfb, *args, **kwargs):
        Names  = beddfa.columns.tolist() + (beddfb.columns + '_i').tolist()
        beddfa = bt.BedTool.from_dataframe(beddfa)
        beddfb = bt.BedTool.from_dataframe(beddfb)
        beddfa = beddfa.intersect(beddfb, **kwargs)\
                    .to_dataframe(disable_auto_names=True,  header=None, names=Names)
        return beddfa

    @emptyfile
    def Peakannotate(self, inbed):
        self.hg38gtf  = self.arg.gtf
        self.annotatepeak = self.arg.annopeak

        cmd = '''
        INBED={inbed}
        GTF={hg38gtf}
        {annotatepeak} \
            $INBED \
            hg38 \
            -gtf $GTF \
            -gid \
            -cpu 10 \
            -go $INBED.go \
            -genomeOntology  $INBED.go.detail \
            1> $INBED.annotate.txt
            2> $INBED.annotate.log'''.format(inbed=inbed, hg38gtf=self.hg38gtf, annotatepeak=self.annotatepeak)
        print(cmd)
        os.system(cmd)

    def InterV2(self, intervals):
        """
        :param intervals: List[List[int]]
        :return: List[List[int]]
        """
        intervals = sorted(intervals)
        if  intervals[0][1] < intervals[1][0]:
            return intervals
        else:
            return [[ intervals[0][0], max(intervals[0][1], intervals[1][1])]]

    def InterVs(self, intervals):
        """
        :param intervals: List[List[int]]
        :return: List[List[int]]
        """
        Inters = sorted(intervals, key=lambda x:x[0])
        merged = [ copy.deepcopy(Inters[0]) ]
        for interval in Inters[1:]:
            if  merged[-1][-1] < interval[0]:
                merged.append(interval)
            else:
                merged[-1][-1] = max(merged[-1][-1], interval[-1])
        return merged

    def MeanCoverDepth(self, Q, R, ignoreOver=True):
        _L = R[1] - R[0] + 1
        _N = np.zeros(_L, int)
        for i in Q:
            for j in self.InterV2(i):
                _N[j[0]-R[0]:j[1]-R[0]+1] += 1

        Cover = len(_N[_N>0])/len(_N)
        Depth = sum(_N)/len(_N)
        return [Cover, Depth]
