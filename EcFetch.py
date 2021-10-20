#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pysam
from joblib import Parallel, delayed
import pandas as pd
import numpy as  np
import os
import re

from concurrent import futures

from EcBammanage import BamFilter
from .EcUtilities import Utilities
from .EcVisual import Visal

class SoftFetch():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.chrs =[str(i) for i in range(1,23)] + ['MT','X','Y'] #remove
        self.minsoflt = self.arg.minsoftdrop
        self.mapq     = self.arg.minmapQ
        self.indel    = self.arg.maskindel
        self.skip     = self.arg.maskskip
        self.hard     = self.arg.maskhard
        self.pad      = self.arg.maskpad
        self.lensoftfq = self.arg.lensoftfq
        self.softfq   = self.arg.getsoftfq

    def _getbam(self, _info):
        if 'bamfile' in _info.keys():
            self.inbam = _info.bamfile
        else:
            self.inbam = '%s/%s.sorted.bam'%(self.arg.Bam, _info.sampleid)
            if not os.path.exists(self.inbam):
                self.inbam = '%s/%s/%s.sorted.bam'%(self.arg.Bam, _info.sampleid, _info.sampleid)
        if not os.path.exists(self.inbam):
            self.log.CW('The bam file of Sample %s cannot be found. Please input right path'%_info.sampleid)
        else:
            return self

    def _getinfo(self, _info):
        self._getbam(_info)
        self.info = _info
        self.inid = _info.sampleid
        self.outdir= '%s/%s'%(self.arg.Fetch, self.inid )
        self.arg.outpre= '%s/%s'%(self.outdir, self.inid)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _transcigar(self, _st):
        CICAR = {'M':0, 'I':1, 'D':2, 'N':3, 'S':4, 'H':5, 'P':6, '=':7, 'X':8}
        _tuple = zip(map(lambda x: CICAR[x], re.split('[0-9]+', _st)[1:]),
                    map(int, re.split('[A-Z=]' , _st)[:-1]))
        return list(_tuple)

    def cigarmerge(self, _ct, indel=100000, skip=1000000000, hard=100000, pad=1000000000, match='Q'):
        indel = self.indel #100000
        skip  = self.skip  #10000000
        hard  = self.hard  #100000
        pad   = self.pad   #10000000

        ct  = [ (0, i[1]) if (i[0] in [7,8]) else i for i in _ct ]
        ctN = ct[:1]
        for i in ct[1:]:
            if match=='Q':
                if ( i[1] <=indel and i[0] ==1 ):
                    i = (0, i[1])
                elif ( i[1] <=indel and i[0] ==2 ):
                    i = (0, 0)
                elif ( i[1] <=skip and i[0] ==3 ):
                    i = (0, 0)
                elif ( i[1] <=hard and i[0] ==5 ):
                    i = (0, 0)
                elif ( i[1] <=pad and i[0] ==6 ):
                    i = (0, 0)
                elif ( i[0] in [7, 8] ):
                    i = (0, i[1])
                else:
                    i = i

            elif match=='R':
                if ( i[1] <=indel and i[0] ==1 ):
                    i = (0, 0)
                elif ( i[1] <=indel and i[0] ==2 ):
                    i = (0, i[1])
                elif ( i[1] <=skip and i[0] ==3 ):
                    i = (0, i[1])
                elif ( i[1] <=hard and i[0] ==5 ):
                    i = (0, 0)
                elif ( i[1] <=pad and i[0] ==6 ):
                    i = (0, 0)
                elif ( i[0] in [7, 8] ):
                    i = (0, i[1])
                else:
                    i = i

            if ( ctN[-1][0]==0 and i[0]==0 ):
                ctN[-1] = (ctN[-1][0] + i[0], ctN[-1][1] + i[1])
            else:
                ctN.append(i)
        return ctN

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

    def _getsoftregion(self, _g):
        #start  = int(_g.reference_start)
        #cigref = eval(_g.cigartuples_ref) if type(_g.cigartuples_ref)==str else _g.cigartuples_ref
        #cigartuples = eval(_g.cigartuples) if type(_g.cigartuples)==str else _g.cigartuples
        #match_type  = ''.join([ 'S' if i[0]==4 else 'M' for i in cigartuples ])

        cigref = _g.cigartuples_ref
        cigartuples = _g.cigartuples

        if _g.is_reverse=='-':
            cigartuples = cigartuples[::-1]

        cigarpos = [(0, cigartuples[0][1])]
        for i in cigartuples[1:]:
            cigarpos.append( (cigarpos[-1][1], cigarpos[-1][1]+i[1]) )
        if _g.is_reverse=='-':
            cigarpos = cigarpos[::-1]

        start  = _g.reference_start
        Regs  = []
        for  n,i in enumerate(cigref):
            if (i[0] !=4 ):
                bed = [ _g.reference_name, start, start + int(i[1]-1), _g.SID, i[1], _g.is_reverse, _g.query_name, 
                        _g.flag, _g.mapping_quality, _g.cigarstring, cigarpos, cigarpos[n], _g.query_length, cigref, _g.query_counts]
                Regs.append(bed)
                start += int(i[1]-1)
        return Regs

    def bamcigarsoft(self):
        samfile = pysam.AlignmentFile(self.inbam, "rb")
        sampd = []
        samfa = {}
        for read in samfile.fetch():
            tags = dict(read.tags)
            raw_cigarstring = read.cigarstring
            raw_cigartuples = read.cigartuples
            reference_name  = read.reference_name
            softlens = [ i[1] for i in raw_cigartuples if (i[0]==4) and (i[1] >self.minsoflt) ]

            if (('SA' in tags) \
                    and ('S' in raw_cigarstring) \
                    and len(softlens) >0) \
                    and read.mapping_quality >=self.mapq :
                    #and (reference_name in self.chrs) \
                #TP = tags['tp'] if 'tp' in tags else ''
                SA = tags['SA']
                query_name = read.query_name
                flag = read.flag
                mapping_quality  = read.mapping_quality 
                reference_start  = read.reference_start
                read.cigartuples = self.cigarmerge(raw_cigartuples, match='Q')
                cigartuples_ref  = self.cigarmerge(raw_cigartuples, match='R')
                query_sequence   = read.query_sequence
                #query_qualities  = pysam.array_to_qualitystring(read.query_qualities)
                is_reverse = '-' if read.is_reverse else '+'
                sampd.append([self.inid, query_name, flag, reference_name, reference_start, mapping_quality, 
                              read.cigartuples, cigartuples_ref, is_reverse, len(query_sequence), SA, 
                              read.cigarstring ,raw_cigarstring])
                if query_name not in samfa:
                    samfa[query_name] = query_sequence
        samfile.close()

        Utilities(self.arg, self.log).savefa(samfa, '%s/%s.chimeric.fasta.gz'%(self.outdir, self.inid))
        del samfa

        sampd = pd.DataFrame(sampd, columns=['SID', 'query_name', 'flag', 'reference_name', 'reference_start', 
                                             'mapping_quality', 'cigartuples', 'cigartuples_ref', 'is_reverse', 
                                             'query_length', 'SA', 'cigarstring', 'raw_cigar'])
        sampd = sampd.merge(sampd.groupby('query_name')['query_name'].size().reset_index(name='query_counts'),
                            on='query_name', how='outer')
        sampd = sampd[(sampd.query_counts>1)]
        sampd.to_csv('%s/%s.chimeric.gz'%(self.outdir, self.inid), sep='\t', index=False)
        return sampd

    def getsoftfq(self, INsoft, softminlen=100):
        softminlen = self.lensoftfq
        f=open( '%s/%s.chimeric.fq'%(self.outdir, self.inid),'w')
        for _n, _l in INsoft.iterrows():
            cigt  = eval(_l.cigartuples) if type(_l.cigartuples)==str else _l.cigartuples
            start = 0
            SEQs  = []
            for  n,i in enumerate(cigt):
                end = int(start) + int(i[1])
                if (i[0] ==4 and i[1]>=softminlen):
                    seq = _l.query_sequence[start:end]
                    #qul = _l.query_qualities[start:end]
                    qul = '~'*len(seq)
                    name= '@%s_soft%s-%s_%s_%s'%(_l.query_name, _l.reference_name, _l.reference_start, n, i[1])
                    SEQs.append('%s\n%s\n+\n%s'%(name, seq, qul) )
                start += int(i[1])
            f.write('\n'.join(SEQs))
        f.close()

    def GetSoft1(self, _info, softfq=False, researh=True):
        softfq = self.softfq
        self._getinfo(_info)
        self.log.CI('start finding soft signal: ' + self.inid)

        ## get bamcigarsoft
        KeepCol = ['SID', 'query_name', 'flag', 'reference_name', 'reference_start', 'mapping_quality', 'cigartuples',
                    'cigartuples_ref', 'is_reverse', 'query_length', 'cigarstring', 'query_counts']
        if researh:
            INsoft = self.bamcigarsoft()[KeepCol]
            INsoft['reference_start'] = INsoft['reference_start'].astype(int)
        else:
            INsoft = pd.read_csv('%s/%s.chimeric.gz'%(self.outdir, self.inid), sep='\t', low_memory=False)[KeepCol]
            INsoft['reference_start'] = INsoft['reference_start'].astype(int)
            INsoft['cigartuples_ref'] = INsoft.cigartuples_ref.map(eval)
            INsoft['cigartuples']     = INsoft.cigartuples.map(eval)

        if INsoft.empty:
            self.log.CW(self.inid +' donot find soft signal...')

        ## get getsoftregion
        BEDS = Parallel( n_jobs=-1, backend='loky')( delayed( self._getsoftregion )(_l) for _n, _l in INsoft.iterrows() )

        if softfq :
            self.getsoftfq(INsoft)
        del(INsoft)

        colm = ['#chrom', 'start', 'end',  'SID', 'length', 'forword', 'query_name', 'flag', 'mapping_quality',
                'cigarstring', 'cigarpos', 'cigarreg' , 'query_length', 'cigarreffilt', 'query_counts']
        BEDS = pd.DataFrame( np.reshape( np.array(BEDS),(-1,len(colm))), columns=colm)
        BEDS.sort_values(by=['#chrom', 'start', 'end', 'query_name', 'cigarreg'],
                            ascending=[True]*5, inplace=True)
        BEDS.to_csv('%s/%s.chimeric.bed'%(self.outdir, self.inid), header=True, index=False, sep='\t')
        Visal().query_length(BEDS , '%s/%s.chimeric.query_length.pdf'%(self.outdir, self.inid))

        self.log.CI('finish finding soft signal: ' + self.inid)

    def bamfetchread(self):
        samfile = pysam.AlignmentFile(self.inbam, "rb")
        #with futures.ThreadPoolExecutor() as executor: #ThreadPoolExecutor/ProcessPoolExecutor
        #    argsmap = ((samfile, _l.chrom, _l.start, _l.end) for _n, _l in chrombin.iterrows())
        #    CouBase = executor.map(self.countbase, argsmap)
        #    CouBase = pd.DataFrame(CouBase, columns= ['chrom', 'start', 'end', 'counts', 'bases'])
        #CouBase = chrombin.merge(CouBase, on=['chrom', 'start', 'end'], how='outer')
        #samfile.close()
        #return CouBase
        def filter_read(read):            
            return not (read.is_secondary
                        or read.is_unmapped
                        or read.is_duplicate
                        or read.query_length - read.query_alignment_length < self.minsoflt
                        or read.is_qcfail
                        or read.mapping_quality < self.mapq)
        sampd = []
        for read in samfile.fetch():
            if filter_read(read):
                raw_cigartuples = read.cigartuples         
                is_reverse = '-' if read.is_reverse else '+'
                Info = [read.reference_name, read.reference_start, read.reference_end, self.inid, 
                        read.reference_end - read.reference_start + 1, is_reverse, read.query_name, 
                        read.query_length, read.query_alignment_length, read.mapping_quality,  read.flag ]
                if is_reverse =='-':
                    read.cigartuples = raw_cigartuples[::-1]
                Info.append(tuple([read.qstart, read.qend]))

                read.cigartuples = self.cigarmerge(raw_cigartuples, match='Q')
                Info.append( read.cigarstring )

                sampd.append(Info)
        samfile.close()


        Head  = ['#chrom', 'start', 'end', 'SID', 'length', 'forword', 'query_name', 
                 'query_length', 'query_alignment_length', 'mapping_quality', 'flag', 
                 'cigarreg',  'cigarstring']
        sampd = pd.DataFrame(sampd, columns=Head)
        
        sampd = sampd.merge(sampd.groupby('query_name')['query_name'].size().reset_index(name='query_counts'),
                            on='query_name', how='outer')
        sampd = sampd[(sampd.query_counts>1)]
        sampd.to_csv( self.arg.outpre + '.chimeric.gz', sep='\t', index=False)
        return sampd

    def GetSoft(self, _info, softfq=False, researh=True):
        softfq = self.softfq
        self._getinfo(_info)
        self.log.CI('start finding soft signal: ' + self.inid)

        samob = BamFilter(vars(self.arg)).bamfilter(self.inbam, self.inid)
        sampd = pd.DataFrame(samob.samlist, columns=samob.Head)

        sampd = sampd.merge(sampd.groupby('query_name')['query_name'].size().reset_index(name='query_counts'),
                            on='query_name', how='outer')
        sampd = sampd[(sampd.query_counts>1)]
        sampd['mapped_reads'] = samob.Counts

        sampd.to_csv( self.arg.outpre + '.chimeric.gz', sep='\t', index=False)
        Visal().query_length(sampd , '%s/%s.chimeric.query_length.pdf'%(self.outdir, self.inid))

        if sampd.empty:
            self.log.CW(self.inid +' donot find soft signal...')
        self.log.CI('finish finding soft signal: ' + self.inid)