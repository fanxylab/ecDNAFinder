#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pysam
from Bio.Seq import Seq

class GetSimu():
    def __init__(self):
        pass
    def simuget(self, reffa, region, ratio, overlap):
        Type = region[-1]
        if Type=='a':
            regions= reffa.fetch(*region[0:3])
            regionn= '%s:%s_%s-r%s-v%s-t%s'%( tuple(region[0:3] + [ratio, overlap, Type]) )
        elif Type=='b':
            regions= reffa.fetch(*region[0:3]) + reffa.fetch(*region[3:6])
            regionn= '%s:%s_%s_%s:%s_%s-r%s-v%s-t%s'%( tuple(region[0:6] + [ratio, overlap, Type]) )
        elif Type=='c':
            regionf= reffa.fetch(*region[0:3])
            regionr= str(Seq(regionf).reverse_complement())[: int((region[2]-region[1])*region[3]) ]
            regions= regionf + regionr
            regionn= '%s:%s_%s_R%s-r%s-v%s-t%s'%( tuple(region[0:4] + [ratio, overlap, Type]) )

        regionb= int(len(regions)*ratio)
        regionq= regions[regionb-overlap:] + regions[:regionb+overlap]
        return '@%s\n%s\n+\n%s'%(regionn, regionq, '~'*len(regionq) )

    def simufa(self):
        hg38_ens='/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
        hg38_ens=pysam.FastaFile(hg38_ens)
        simatx=[
            ['10', 13070000, 13090000, 'a'], 
            ['11', 13800000, 13830000, 'a'],
            ['1' , 13610000, 13615000, '3', 13710000, 13715000, 'b'],
            ['2' , 67250000, 67251000, '2', 67800000, 67810000, 'b'], 
            ['3' , 80950000, 80970000, 0.2, 'c'],
            ['3' , 80950000, 80970000, 0.5, 'c'],
            ['3' , 80950000, 80970000, 0.7, 'c'],
            ['3' , 80950000, 80970000, 1.0, 'c'],
            ['4' , 21500000, 21520000, 0.2, 'c'],
            ['4' , 21500000, 21520000, 0.5, 'c'],
            ['4' , 21500000, 21520000, 0.7, 'c'],
            ['4' , 21500000, 21520000, 1.0, 'c'],
            #['12', 13830000, 13800000, 'a1'],
            #['13', 13830000, 13800000, 'a2'],
            #['3' , 67250000, 67251000, '2', 67800000, 67810000, 'b1'], 
            #['4' , 67250000, 67251000, '2', 67800000, 67810000, 'b2'], 
        ]
        ratios=[0.1, 0.3, 0.5, 0.7, 0.9]
        ovlaps=[0, 100, -100, -500]
        allfas=[ self.simuget(hg38_ens, _s, _r, _o) for _s in simatx for _r in ratios for _o in ovlaps]
        allfas='\n'.join(allfas)
        f=open('./simulate.fq','w')
        f.write(allfas)
        f.close()
#GetSimu().simufa()
