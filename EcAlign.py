#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os

class Align():
    def __init__(self, fq1, inid, outdir):
        self.fq1 = fq1
        self.inid = inid
        self.outdir=outdir + '/' + inid
        os.makedirs(self.outdir, exist_ok=True)

    def emptyfile(_func):
        def wrapper(self, *args, **kargs):
            file = self.fq1
            if os.path.exists(file):
                sz = os.path.getsize(file)
                if not sz:
                    print(file, " is empty!")
                else:
                    _func(self, *args, **kargs)
            else:
                print(file, " is not exists!")
        return wrapper

    @emptyfile
    def SoftMINI(self,
                 samtls='/share/home/share/software/samtools-1.10/bin/samtools',
                 minip2='/share/home/share/software/minimap2-2.17_x64-linux/minimap2',
                 REF='/share/home/share/Repository/GenomeDB/Index/Homo_Sapiens/MinimapDB/ENSEMBL_GRch38_PRIM_MINIM/Homo_sapiens.GRCh38.dna.primary_assembly.mmi'):
        cmd = '''
        mkdir -p {outdir}
        {minimap2} \
            -ax asm20 \
            -t 20 \
            -k 19  -w 10 -H -O 5,56 -E 4,1 -A 2 -B 5 -z 400,50 -r 2000 -g 5000 --lj-min-ratio 0.5 \
            --cs --MD -Y \
            --secondary no \
            {REF} \
            {fq1} | \
            {samtools} view -bS -@ 10  - \
            > {outdir}/{ID}.bam
        {samtools} sort -@ 20 -o {outdir}/{ID}.sorted.bam {outdir}/{ID}.bam
        {samtools} index -@ 20  {outdir}/{ID}.sorted.bam
        '''.format(minimap2=minip2, REF=REF, fq1=self.fq1, samtools=samtls,outdir=self.outdir,ID=self.inid)
        print(cmd)
        os.system(cmd)
