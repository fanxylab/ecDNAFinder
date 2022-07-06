#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import multiprocessing as mp
import pysam

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from concurrent import futures


chr_list = [str(i) for i in range(1,23)] + ['X', 'Y']


class Index:
    n_gap_file = '/share/home/share/Repository/GenomeDB/Blacklist/hg38_ucsc_track_gap.txt'
    genome_file = '/share/home/zhou_wei/Workspace/01Repository/GenomeDB/Reference/EcDNARef/HG38_ENSEMBL_Plasmid20.fa'
    def __init__(self, bin_size=5000, bin_min_size=0.65,
                 merge_size=500000, merge_min_size=0.45):
        self.dropbinlen = 1000
        self.bin_size = bin_size
        self.bin_min_size = bin_min_size
        self.merge_size = merge_size
        self.merge_min_size = merge_min_size
        self.ref = pysam.FastaFile(self.genome_file)

    def build_index(self):
        self.drop_n_gaps()
        self.split_bin_continuous()

    def drop_n_gaps(self):
        n_gaps = {x:[] for x in chr_list}
        with open(self.n_gap_file) as f:
            for line in f:
                _bin, _chr, _start, _end, _ix, _n, _size, _type, _bridge = line.strip().split('\t')
                _chr = _chr.lstrip('chr')
                if _chr not in chr_list:
                    continue
                n_gaps[_chr].append([int(_start), int(_end)])
        genome_idx = {}
        with open(self.genome_file + '.fai') as f:
            for line in f:
                try:
                    _chr, _end, _offset, _linebases, _linewidth = line.strip().split('\t')
                except:
                    print(line)
                if _chr not in chr_list:
                    continue
                genome_idx[_chr] = [0, int(_end)]
        self.n_gap_chrom = {}
        for _chr in chr_list:
            self.n_gap_chrom[_chr] = self.cal_complement(genome_idx[_chr], n_gaps[_chr])

    def cal_complement(self, genome_idx, n_gaps):
        n_gaps = sorted(n_gaps, key=lambda x: (x[0], x[1]))
        start = [genome_idx[1]] + [x[1] for x in n_gaps]
        end = [x[0] for x in n_gaps] + [genome_idx[1]]
        return [[s, e, e - s] for s, e in zip(start, end) if e - s > self.dropbinlen]

    def divide_bin_continuous(self, _chr, bins, size, min_bin_ratio):
        '''
        根据bin生成size区间，保证bin个数和长度
        '''
        region = [bins[0][0]]
        last_bin_size = 0
        for _s, _e, _l in bins:
            ns = _s + size - last_bin_size
            if size >= _e - _s + last_bin_size:
                last_bin_size += _e - _s
                continue
            region.extend(range(ns, _e, size))
            last_bin_size = _e - region[-1]
        if len(region) == 1 or last_bin_size / size > min_bin_ratio:
            region.append(bins[-1][1])
        else:
            region[-1] = bins[-1][1]
        result = []
        for x, y in zip(region[:-1], region[1:]):
            seq = self.ref.fetch(_chr, x, y)
            length = len(seq)
            length -= seq.count('N') + seq.count('n')
            try:
                gc = sum(seq.count(x) for x in 'GCgc') / length
                rmsk = sum(seq.count(x) for x in 'atcg') / length
            except ZeroDivisionError:
                gc, rmsk = 0.0, 0.0
            result.append([x, y, gc, rmsk])
        return result


    def split_bin_continuous(self):
        with open('sBin.txt', 'w') as f:
            for _chr, bins in self.n_gap_chrom.items():
                sbins = self.divide_bin_continuous(_chr, bins, self.bin_size, self.bin_min_size)
                for sb in sbins:
                    f.write('%s\t%s\n' % (_chr, '\t'.join(str(x) for x in sb)))
        with open('mBin.txt', 'w') as f:
            for _chr, bins in self.n_gap_chrom.items():
                mbins = self.divide_bin_continuous(_chr, bins, self.merge_size, self.merge_min_size)
                for mb in mbins:
                    f.write('%s\t%s\n' % (_chr, '\t'.join(str(x) for x in mb)))


class CNV:
    def __init__(self, sample_name, input_dir, out_dir):
        self.sample_name = sample_name
        self.load_bam(input_dir)
        self.out_dir= out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.s_bin = self.load_index(self.script_dir + os.sep + 'sBin.txt')
        self.m_bin = self.load_index(self.script_dir + os.sep + 'mBin.txt')
        self.result_file = '%s/%s.countsbases.txt' % (self.out_dir, self.sample_name)

    def load_bam(self, bam_dir='bam'):
        inbam = '%s/%s.sorted.bam'% (bam_dir, self.sample_name)
        if not os.path.exists(inbam):
            raise Exception('The bam file of Sample %s cannot be found. Please input right path' % sample_name)
        self.samfile = pysam.AlignmentFile(inbam, 'rb')

    def load_index(self, idx_file):
        _index = {x:[] for x in chr_list}
        with open(idx_file) as f:
            for line in f:
                _chr, s, e, gc, rmsk = line.strip().split('\t')
                _index[_chr].append([int(s), int(e), float(gc), float(rmsk)])
        return _index

    def count_base(self, _bin, _file):
        with open(_file, 'w') as f:
            for _chr, b in _bin.items():
                for _start, _end, gc, rmsk in b:
                    _chr, _start, _end, count, bases = self.count_base_in_region(_chr, _start, _end)
                    f.write('%s\n' % '\t'.join(str(x) for x in [_chr, _start, _end, gc, rmsk, self.sample_name, count, bases]))

    def count_base_in_chrom(self, _chr):
        print("%s start:" % _chr, time.time())
        with open('%s.ch%s.bin.count_bases.txt' % (self.out_dir + os.sep + self.sample_name, _chr), 'w') as f:
            for _start, _end, m_gc, m_rmsk in self.m_bin[_chr]:
                _chr, _start, _end, m_count, m_bases = self.count_base_in_region(_chr, _start, _end)
                e_info = []
                for s, e, e_gc, e_rmsk in self.s_bin[_chr]:
                    if s < _start or e > _end:
                        continue
                    _chr, s, e, e_count, e_bases = self.count_base_in_region(_chr, s, e)
                    e_info.append([e_count, e_bases])
                e_count, e_bases = np.mean(e_info, axis=0)
                f.write('%s\n' % '\t'.join(str(x) for x in [_chr, _start, _end, m_gc, m_rmsk, m_count, m_bases, e_count, e_bases]))
        print("%s finish:" % _chr, time.time())

    def count_base_parallel(self):
        processes = []
        for _chr in self.m_bin:
            p = mp.Process(target=self.count_base_in_chrom, args=(_chr,))
            processes.append(p)
        [x.start() for x in processes]
        [x.join() for x in processes]

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
            return 0, 0
        count = 0
        bases = 0
        for read in bamfetch:
            if filter_read(read):
                # Only count the bases aligned to the region
                rlen = read.query_alignment_length
                if read.reference_start < _start:
                    rlen -= _start - read.reference_start
                if read.reference_end > _end:
                    rlen -= read.reference_end - _end
                # TODO(enze): 既然匹配位置和匹配长度之间没关系，那两种计算方式都是错的
                #tlen = min(_end, read.reference_end) - max(_start, read.reference_start)
                rlen = abs(rlen) # del len lead to negative value
                if rlen >= min_lenc:
                    bases += rlen
                    count += 1
        return _chr, _start, _end, count, bases

    def core_segment(self):
        header = ['chrom', 'start', 'end', 'gc', 'rmsk', 'counts', 'bases', 'counts_md','bases_md']
        header_cmd = 'echo "%s" > %s' % ('\t'.join(header), self.result_file)
        merge_cmd = 'cat `ls %s/%s.ch* | sort -V` >> %s' % (self.out_dir, self.sample_name, self.result_file)
        cnv_cmd = '/datd/enzedeng/software/R-4.1.1/bin/Rscript %s/ezCNV.R %s %s' % (self.script_dir, self.out_dir, self.result_file)
        os.system(header_cmd)
        os.system(merge_cmd)
        os.system(cnv_cmd)

    def do_coverage(self):
        if not os.path.exists(self.result_file):
            self.count_base_parallel()
        self.core_segment()


if __name__ == '__main__':
    #idx = Index()
    #idx.build_index()
    if len(sys.argv) < 4:
        print("Usage: %s sample_name input_dir output_dir" % sys.argv[0])
        sys.exit()
    sample_name, input_dir, output_dir = sys.argv[1:]
    c = CNV(sample_name, input_dir, output_dir)
    c.do_coverage()
