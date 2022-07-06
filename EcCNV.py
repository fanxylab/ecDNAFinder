#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import pysam
from Bio.Seq import Seq

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from concurrent import futures
#from .EcVisual import Visal

#sklearn.neighbors.KernelDensity CNVnator
#cnvkit.py access \
#    /share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/UCSC/HG38/hg38_ucsc.fa \
#    -o hg38.bed
# 使用cnvkit access 进行splitbed

class BinBuild():
    def __init__(self, arg=None, log=None):
        self.arg = arg
        self.log = log
        self.ngaps = self.arg.ngaps
        self.blacklist = self.arg.blacklist
        self.dropbinlen = self.arg.dropbinlen
        self.genome = self.arg.genomecnv
        self.cytoband = self.arg.cytoband

    def Ccmplmty(self, interVs, Min=0, Max=40):
        interVs= interVs[np.lexsort((interVs[:,1], interVs[:,0]))]

        End = np.append(interVs[:,0], Max)
        Str = np.insert(interVs[:,1], 0, Min)
        INt = np.c_[Str, End]

        INt = INt[(INt[:,1]-INt[:,0]) >0]
        return INt

    def divbin_continu(self, Inter, Sep, info=[], endown=0.65):
        Split = []
        Extra = 0
        for _s, _e in Inter:
            if Extra % Sep == 0:
                region = range(_s, _e, Sep)
                Split.extend(region)
                Extra = _e - region[-1]
            else:
                _S = _s + Sep - Extra
                if _S >= _e:
                    Extra += _e - _s
                else:
                    region = range(_S, _e, Sep)
                    Split.extend(region)
                    Extra = _e - region[-1]
        if len(Split) ==1 or Extra/Sep > endown:
            Split.append(Inter[-1][-1])
        else:
            Split[-1] = Inter[-1][-1]
            Extra += Sep

        #Split = [ info + Split[i] + [i+1] for i in range(len(Split) - 1) ]
        Split = [ info + [Split[i], Split[i+1], Sep, i+1] for i in range(len(Split) - 1) ]
        Split[-1][-2] = Extra
        return Split

    def GC_Rmsk(self, Seq, Ctype='GCgc', InN=False): #'atcg'
        '''
        bedtools nuc -fi bed
        '''
        Len = len(Seq)
        if not InN:
            Len -= Seq.count('N')+Seq.count('n')
        try:
            return sum(Seq.count(x) for x in Ctype )/Len
        except ZeroDivisionError:
            return 0.0

    def DropNGaps(self):
        Ngaps = pd.read_csv(self.ngaps, sep='\t', comment=None)[['chrom', 'chromStart', 'chromEnd', 'type']]
        Ngaps['chrom'] = Ngaps['chrom'].str.lstrip('chr')

        GFai  = pd.read_csv(self.genome + '.fai', sep='\t',
                            names=['chrom', 'end', 'offset', 'linebases', 'linwidth'],
                            comment=None)
        GFai['start'] = 0

        KeepCol=['chrom', 'start', 'end']
        NChrom = []
        for (_c, _s, _e), _g in GFai[KeepCol]\
                                    .merge(Ngaps, how='left', on='chrom')\
                                    .groupby(by=KeepCol):
            if  not _g.type.isna().any():
                _g = pd.DataFrame(self.Ccmplmty(_g[['chromStart', 'chromEnd']].values, Min=_s, Max=_e),
                                    columns=KeepCol[1:3])
                #_g.insert(0, 'chrom', _c)
                _g['chrom'] = _c

            NChrom.append(_g[KeepCol])
        NChrom = pd.concat(NChrom, axis=0, sort=False)
        NChrom[KeepCol[1:3]] = NChrom[KeepCol[1:3]].astype(int)
        NChrom['length'] = NChrom.end - NChrom.start
        NChrom = NChrom[(NChrom.length > self.dropbinlen)]
        return NChrom

    def SpltBin_continu(self, Inbed):
        Ref = pysam.FastaFile(self.genome)

        mbin = self.arg.mergebin
        emer = self.arg.endmergepfre

        sbin = self.arg.splitbin
        eown = self.arg.endindepfre

        Inbed = Inbed.sort_values(by=['chrom', 'start', 'end'])

        SBin = [ self.divbin_continu(_g[['start', 'end']].values, sbin, info=[_c], endown=eown) for _c, _g in Inbed.groupby(by='chrom')]
        SBin = pd.DataFrame( sum(SBin, []), columns=['chrom', 'eS', 'eE', 'eL', 'ebin'])
        SBin['eGC']   = SBin.apply(lambda x: self.GC_Rmsk(Ref.fetch(x.chrom, x.eS, x.eE)), axis=1)
        SBin['eRmsk'] = SBin.apply(lambda x: self.GC_Rmsk(Ref.fetch(x.chrom, x.eS, x.eE), Ctype='atcg'), axis=1)
        SBin.to_csv("SBin2.txt", sep='\t', header=False, index=False, columns=['chrom', 'eS', 'eE', 'eGC', 'eRmsk'])

        MBin = [ self.divbin_continu(_g[['start', 'end']].values, mbin, info=[_c], endown=emer) for _c, _g in Inbed.groupby(by='chrom')]
        MBin = pd.DataFrame( sum(MBin, []), columns=['chrom', 'mS', 'mE', 'mL', 'mbin'])
        MBin['mGC']   = MBin.apply(lambda x: self.GC_Rmsk(Ref.fetch(x.chrom, x.mS, x.mE)), axis=1)
        MBin['mRmsk'] = MBin.apply(lambda x: self.GC_Rmsk(Ref.fetch(x.chrom, x.mS, x.mE), Ctype='atcg'), axis=1)
        MBin.to_csv("MBin2.txt", sep='\t', header=False, index=False, columns=['chrom', 'mS', 'mE', 'mGC', 'mRmsk'])

        MBin = pd.concat([ x.to_frame().T.merge(SBin[((SBin.chrom==x.chrom) & (SBin.eS>=x.mS) & (SBin.eE<=x.mE))], on='chrom', how='outer')
                            for _, x in MBin.iterrows()], axis=0)
        MBin.to_csv("MBin0.txt", sep='\t', header=False, index=False)

        return MBin

    def NoNBed(self):
        NChrom = self.DropNGaps()
        #NChrom = NChrom[NChrom['chrom']=='Y']
        NChrom.to_csv("NChrom1.txt")
        MChrom = self.SpltBin_continu(NChrom)
        NChrom.to_csv("NChrom2.txt")
        MChrom.to_csv(self.arg.buildidx, sep='\t', index=False)
        for _i in ['eGC', 'mGC']:
            if _i.startswith('m'):
                Dup=['chrom', 'mS', 'mE', 'mL']
            else:
                Dup=[]
            #Visal().GCdistribut(MChrom, '%s%s.pdf'%(self.arg.buildidx.rstrip('idx'), _i), X=_i, Dup=Dup)

class Correct():
    def __init__(self, arg, log,  *array, **dicts):
        self.arg = arg
        self.log = log
        self.array  = array
        self.dicts  = dicts

    def Rcmds(self):
        self.gcdelfi = '''
        lowess.gc <- function(coveragA, biaA) {
            NONAIdx  <- which(! is.na(coveragA))
            coverage <- coveragA[NONAIdx]     ## excluse NA Inf value in  coverage
            coveragB <- rep(NA, length(coveragA))
            bias     <-biaA[NONAIdx]

            i <- seq(min(bias, na.rm=TRUE), max(bias, na.rm=TRUE), by = 0.001)
            coverage.trend <- loess(coverage ~ bias)
            coverage.model <- loess(predict(coverage.trend, i) ~ i)
            coverage.pred <- predict(coverage.model, bias)
            coverage.corrected <- coverage - coverage.pred + median(coverage)
            coveragB[NONAIdx] <- coverage.corrected
            return (coveragB)
        }
        '''
        self.gcginkgo='''
        lowess.gc <- function(jtkx, jtky) {
                jtklow <- lowess(jtkx, log(jtky), f=0.05)
                jtkz <- approx(jtklow$x, jtklow$y, jtkx)
                return(exp(log(jtky) - jtkz$y))
        }
        '''
        self.gchmmc='''
        lowess.gc <- function(x, mappability = 0.9, samplesize = 50000,
                                    routlier = 0.01, doutlier = 0.001,
                                    coutlier <- 0.01,
                                    verbose = TRUE) {
        x$valid <- TRUE
        x$valid[x$reads <= 0 | x$gc < 0] <- FALSE
        x$ideal <- TRUE

        range <- quantile(x$reads[x$valid], prob = c(0, 1 - routlier), na.rm = TRUE)
        domain <- quantile(x$gc[x$valid], prob = c(doutlier, 1 - doutlier),na.rm = TRUE)

        x$ideal[!x$valid | x$map < mappability | x$reads <= range[1] |
                    x$reads > range[2] | x$gc < domain[1] | x$gc > domain[2]] <- FALSE

        set <- which(x$ideal)
        select <- sample(set, min(length(set), samplesize))
        rough = loess(x$reads[select] ~ x$gc[select], span = 0.03)
        i <- seq(0, 1, by = 0.001)
        final = loess(predict(rough, i) ~ i, span = 0.3)
        x$cor.gc <- x$reads / predict(final, x$gc)


        range <- quantile(x$cor.gc[which(x$valid)],
                            prob = c(0, 1 - coutlier), na.rm = TRUE)
        set <- which(x$cor.gc < range[2])
        select <- sample(set, min(length(set), samplesize))
        final = approxfun(lowess(x$map[select], x$cor.gc[select]))

        x$cor.map <- x$cor.gc / final(x$map)
        x$copy <- x$cor.map
        x$copy[x$copy <= 0] = NA
        x$copy <- log(x$copy, 2)
        return(x)
        }

        '''

class BinCount():
    def __init__(self, arg, log,  *array, **dicts):
        self.arg = arg
        self.log = log
        self.array  = array
        self.dicts  = dicts
        self.NULL_LOG2_COVERAGE = -20.0
        self.hchrs  = [str(i) for i in range(1,23)] + ['MT','X','Y']
        self.plasmids = ['2x35S-eYGFPuv-T878-p73', '2x35S-LbCpf1-pQD', '380B-eYGFPuv-d11-d15', '380K-eYGFPuv',
                        '380K-eYGFPuv-d123456', '5P2T-pKGW7', 'A10-pg-p221', 'Cas9-U6-sgRNA-pQD', 'd2A-E9t-v4',
                        'HD-T878-UBQ10', 'HG-F2A-pQD', 'Lat52-grim-TE-MC9-prk6-pKGW7', 'Lat52-RG-HTR10-1-GFP-pBGW7',
                        'myb98-genomic-nsc-TOPO', 'pB2CGW', 'pHDzCGW', 'pQD-in', 'pro18-Mal480-d1S-E9t',
                        'SunTag-CRISPRi', 'V7-MC-HG-FA']
        self.chrs = self.hchrs + self.plasmids

    def _getbam(self, _info):
        if 'bamfile' in _info.keys():
            self.inbam = _info.bamfile
        else:
            self.inbam = '%s/%s.rmdup.bam'%(self.arg.Bam, _info.sampleid)
            if not os.path.exists(self.inbam):
                self.inbam = '%s/%s/%s.rmdup.bam'%(self.arg.Bam, _info.sampleid, _info.sampleid)
        if not os.path.exists(self.inbam):
            self.log.CW('The bam file of Sample %s cannot be found. Please input right path'%_info.sampleid)
        else:
            return self

    def _getinfo(self, _info):
        self._getbam(_info)
        self.info = _info
        self.inid = _info.sampleid
        self.outdir= '%s/%s'%(self.arg.CNV, self.inid )
        self.arg.outpre= '%s/%s'%(self.outdir, self.inid)
        self.arg.scriptdir = os.path.dirname(os.path.realpath(__file__))
        self.result_file = '%s/%s.countsbases.txt' % (self.outdir, self.inid)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getchrom(self, ct='m',  minlen=2000):
        COLR =  ['chrom', ct+'S', ct+'E', ct+'L', ct+'GC', ct+'Rmsk', 'mbin']
        COLN =  ['chrom', 'start', 'end', 'length', 'gc', 'rmsk', 'bins']
        if ct=='e':
            COLR.append('ebin')
            COLN.append('ebin')

        chrombin = pd.read_csv(self.arg.buildidx, sep='\t')
        chrombin.chrom   = chrombin.chrom.astype(str)
        chrombin = chrombin.loc[(chrombin.chrom.isin(self.chrs) & (chrombin.mL>=minlen)), COLR].drop_duplicates(keep='first')
        chrombin.columns = COLN
        chrombin['SID']  = self.inid
        return chrombin

    def countbase(self, args): #bamfile, chrom, start, end, min_mapq=0):
        """Calculate depth of a region via pysam count.
        i.e. counting the number of read starts in a region, then scaling for read
        length and region width to estimate depth.
        Coordinates are 0-based, per pysam.
        bedtools coverage
        samtools bedcov
        """
        bamfile, chrom, start, end = args
        min_mapq=0
        min_lenc=10

        def filter_read(read):
            return not (read.is_duplicate
                        or read.is_secondary
                        or read.is_unmapped
                        or read.is_qcfail
                        or read.mapping_quality < min_mapq)

        count = 0
        bases = 0
        try:
            bamfetch = bamfile.fetch(reference=str(chrom), start=start, end=end, multiple_iterators =True)
        except (ValueError, ArithmeticError) :
            print('invalid contig ' + chrom)
            return [chrom, start, end, count, bases]

        for read in bamfetch:
            if filter_read(read):
                # Only count the bases aligned to the region
                rlen = read.query_alignment_length
                if read.reference_start < start:
                    rlen -= start - read.reference_start
                if read.reference_end > end:
                    rlen -= read.reference_end - end
                #rlen = min(end, read.reference_end) - max(start, read.reference_start)
                rlen = abs(rlen) # del len lead to negative value
                if rlen>= min_lenc:
                    bases += rlen
                    count += 1
        return [chrom, start, end, count, bases]

    def bedcov(self, bed_fname, bam_fname, min_mapq):
        """Calculate depth of all regions in a BED file via samtools (pysam) bedcov.
        i.e. mean pileup depth across each region.
        """
        # Count bases in each region; exclude low-MAPQ reads
        cmd = [bed_fname, bam_fname]
        if min_mapq and min_mapq > 0:
            cmd.extend(['-Q', bytes(min_mapq)])
        try:
            raw = pysam.bedcov(*cmd, split_lines=False)
        except pysam.SamtoolsError as exc:
            raise ValueError("Failed processing %r coverages in %r regions. "
                            "PySAM error: %s" % (bam_fname, bed_fname, exc))
        if not raw:
            raise ValueError("BED file %r chromosome names don't match any in "
                            "BAM file %r" % (bed_fname, bam_fname))
        columns = detect_bedcov_columns(raw)
        table = pd.read_csv(StringIO(raw), sep='\t', names=columns, usecols=columns)
        return table

    def Binparall(self, chrombin):
        samfile = pysam.AlignmentFile(self.inbam, "rb")
        with futures.ThreadPoolExecutor() as executor: #ThreadPoolExecutor/ProcessPoolExecutor
            argsmap = ((samfile, _l.chrom, _l.start, _l.end) for _n, _l in chrombin.iterrows())
            CouBase = executor.map(self.countbase, argsmap)
            CouBase = pd.DataFrame(CouBase, columns= ['chrom', 'start', 'end', 'counts', 'bases'])
        CouBase = chrombin.merge(CouBase, on=['chrom', 'start', 'end'], how='outer')
        samfile.close()
        return CouBase

    def CopyRatio(self, countdf):
        countdf['log2_count'] = np.log2( countdf['counts']/countdf['counts'].mean())
        countdf['log2_depth'] = np.log2( (countdf['bases']/countdf['mL'])/(countdf['bases'].sum()/countdf['mL'].sum()))
        return countdf

    def CountBase(self, CT='e'):
        CouBase  = self.Binparall(self._getchrom(ct=CT))
        if CT =='e':
            CouBase.to_csv( '%s.%sbin.countsbases.%s.gz'%(self.arg.outpre, CT, self.arg.binsize), sep='\t', index=False)

        return CouBase

    def DoCoverage(self, _l): # bamfile, bedfile, minmapq=0):
        eCB = self.CountBase(CT='e')
        mCB = self.CountBase(CT='m')

        eCB = eCB.groupby(by=['chrom', 'bins'], sort=False)
        mCB = mCB.merge( eCB['counts'].mean().to_frame('counts_md').reset_index(), on=['chrom', 'bins'], how='outer')\
                 .merge( eCB['bases'].mean().to_frame('bases_md').reset_index(), on=['chrom', 'bins'], how='outer')
        mCB.to_csv( '%s.%sbin.countsbases.%s.gz'%(self.arg.outpre, 'm', self.arg.binsize), sep='\t', index=False)

        #CopyStat= self.CopyRatio(CouBase)
        #CouBase.to_csv( self.arg.outpre + '.Lbin.copyratio.txt', sep='\t', index=False)
    def CorSeg(self, _l):
        cmd = """zcat %s/%s.mbin.countsbases.%s.gz | awk 'NR==1||$1~"^[0-9XY]+$"{OFS="\\t";print $1,$2,$3,$5,$6,$9,$10,$11,$12}' > %s/%s.txt""" % (
            self.outdir, self.inid, self.arg.binsize, self.outdir, self.inid)
        os.system(cmd)
        cnv_cmd = '/datd/enzedeng/software/R-4.1.1/bin/Rscript %s/ecCNV.R %s %s/%s.txt' % (
            self.arg.scriptdir, self.outdir, self.outdir, self.inid)
        os.system(cnv_cmd)
        #CT = 'm'
        #outpre = '%s.%sbin.countsbases.%s'%(self.inid, CT, self.arg.binsize)
        #os.system( '/datd/enzedeng/software/R-4.1.1/bin/Rscript %s/cnvgcnomal.R %s %s &'
        #            %(self.arg.scriptdir, self.outdir, outpre))

    def LinePlt(self, _L):
        '''deprecated'''
        Lbin = [ '{0}/{1}/{1}.Lbin.copyratio.txt'.format(self.arg.CNV, i) for i in _L.sampleid]
        Lbin = pd.concat([ pd.read_csv(i, sep='\t') for i in Lbin], axis=0)
        Lbin.sort_values(by=['chrom', 'mS', 'mE'], inplace=True)
        Lbin['Bins'] = Lbin.chrom + ':' + Lbin.mS.astype(str) + '-' + Lbin.mE.astype(str)
        Lbin['GrpC'] = Lbin.chrom.apply(lambda x: x if x in self.hchrs else 'plasmid')
        Lbin.to_csv(self.arg.CNV + '/CNV.1M.txt',sep='\t', index=False)
        os.system('/share/home/share/software/R-3.6.3/bin/Rscript /datb/zhouwei/01Projects/03ecDNA/Nanopore/cnv.line.plot.R {0}/CNV.1M.txt {0}'.format(self.arg.CNV))
        #Visal().lineplt(Lbin, './aa.pdf')

class CNVpipe():
    def __init__(self, arg, log,  *array, **dicts):
        self.arg = arg
        self.log = log
        self.array  = array
        self.dicts  = dicts
        self.arg.scriptdir = os.path.dirname(os.path.realpath(__file__))
        self.arg.datadir   = self.arg.scriptdir + '/../Data/'
        self.arg.splitstr = str( int(self.arg.splitbin/1e3)
                                    if self.arg.splitbin/1e3 >=1
                                    else round(self.arg.splitbin/1e3, 1) ) + 'K'
        self.arg.mergestr = str(int(self.arg.mergebin/1e3)) + 'K' \
                                    if self.arg.mergebin/1e6 <1 \
                                    else str(round(self.arg.mergebin/1e6, 1)) + 'M'
        self.arg.binsize = self.arg.mergestr + self.arg.splitstr

    def buildidx(self):
        if not self.arg.buildidx:
            self.arg.buildidx = '%s/hg38_split_%s_continue_bin.idx'%(self.arg.datadir, self.arg.binsize)
            self.log.CI('Cannot find the bin file. The defaul file will be used.')
            if not os.path.exists(self.arg.buildidx):
                self.log.CI('Cannot find the defaul bin file. The file will be built.')
                BinBuild( self.arg, self.log ).NoNBed()

    def comptCNV(self, _l):
        bc = BinCount( self.arg, self.log )
        bc._getinfo(_l)
        if not os.path.exists(f'{self.arg.outpre}.mbin.countsbases.{self.arg.binsize}.gz'):
            bc.DoCoverage(_l)
        bc.CorSeg(_l)

    def plotall(self, _L):
        pass

    def integrat(self, _L):
        self.buildidx()
        #self.arg.buildidx = '%s/hg38_split_%s_continue_bin.idx'%(self.arg.datadir, self.arg.binsize)
        Parallel( n_jobs=self.arg.njob, verbose=1 )( delayed( self.comptCNV  )(_l) for _n, _l in _L.iterrows() )
        self.plotall(_L)
