#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import collections
import pandas as pd
import numpy  as np
from joblib import Parallel, delayed
import copy

from EcMagiccube import InterV2, InterVs, InterSm, OrderLinks, GroupBY, trimOverlink
from .EcUtilities import Utilities
from .EcVisual import Visal
from .EcAnnotate import Annotate

class EcRegion():
    def __init__(self, chr, start, end, query_name, cell_name):
        self.chr = chr
        self.start = start
        self.end = end
        self.strand = '+' if start < end else '-'
        self.name = list(set(query_name))
        self.cell = set(cell_name)

    def reverse(self):
        self.start, self.end = self.end, self.start
        self.strand = '+' if self.start < self.end else '-'

    def __lt__(self, other):
        if type(other) != type(self):
            return False
        if self.chr == other.chr:
            return self.start < other.start
        else:
            return self.chr < other.chr

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if self.chr == other.chr and self.start == other.start and self.end == other.end:
            return True
        return False

    def serialize_to_string(self):
        return "\t".join(str(x) for x in [self.chr, self.start, self.end, self.strand, self.name, self.cell])

    def parse_from_string(self, string):
        chr, start, end, strand, name, cell = string.strip().split('\t')
        self.chr = chr
        self.start = int(start)
        self.end = int(end)
        self.strand = strand
        self.name = eval(name)
        self.cell = eval(cell)

    def __repr__(self):
        return f'{self.chr}:{self.start}-{self.end}:{self.strand}'


class BreakPoint():
    def __init__(self, chr1=None, bpt1=None, dir1=None,
                       chr2=None, bpt2=None, dir2=None,
                       query_name='', cell_name='', sort=True):
        bp1 = (chr1, bpt1, dir1)
        bp2 = (chr2, bpt2, dir2)
        if sort:
            bps = sorted([bp1, bp2], key=lambda x: (x[0], x[1], x[2]))
        else:
            bps = [bp1, bp2]
        self.bp1, self.bp2 = bps
        self.chr1 = self.bp1[0]
        self.bpt1 = self.bp1[1]
        self.dir1 = self.bp1[2]
        self.chr2 = self.bp2[0]
        self.bpt2 = self.bp2[1]
        self.dir2 = self.bp2[2]
        if type(query_name) == str:
            self.name = [query_name]
        else:
            self.name = query_name
        if type(cell_name) == str:
            self.cell = {cell_name}
        elif cell_name:
            self.cell = cell_name

    def reverse(self):
        return BreakPoint(self.chr2, self.bpt2, self.dir2, self.chr1, self.bpt1, self.dir1, self.name, self.cell, sort=False)

    def __lt__(self, other):
        if type(other) != type(self):
            return False
        #(x.chr1, x.chr2, x.bpt1, x.bpt2, x.dir1, x.dir2))
        #return any([self.chr1 < other.chr1, self.chr2 < other.chr2,
        #            self.bpt1 < other.bpt1, self.bpt2 < other.bpt2,
        #            self.dir1 < other.dir1, self.dir2 < other.dir2])
        if self.chr1 != other.chr1:
            return self.chr1 < other.chr1
        if self.chr2 != other.chr2:
            return self.chr2 < other.chr2
        if self.bpt1 != other.bpt1:
            return self.bpt1 < other.bpt1
        if self.bpt2 != other.bpt2:
            return self.bpt2 < other.bpt2
        if self.dir1 != other.dir1:
            return self.dir1 < other.dir1
        return self.dir2 < other.dir2

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        if (self.chr1 == other.chr1 and self.bpt1 == other.bpt1 and self.dir1 == other.dir1 and
            self.chr2 == other.chr2 and self.bpt2 == other.bpt2 and self.dir2 == other.dir2):
            return True
        return False

    def serialize_to_string(self):
        return "\t".join(str(x) for x in [self.chr1, self.bpt1, self.dir1, self.chr2, self.bpt2, self.dir2, self.name, self.cell])

    def parse_from_string(self, string):
        chr1, bpt1, dir1, chr2, bpt2, dir2, name, cell = string.strip().split('\t')
        self.chr1 = chr1
        self.bpt1 = int(bpt1)
        self.dir1 = dir1
        self.chr2 = chr2
        self.bpt2 = int(bpt2)
        self.dir2 = dir2
        self.name = eval(name)
        self.cell = eval(cell)

    def __repr__(self):
        return self.serialize_to_string()

def trans_region(region):
    if len(region) == 1:
        bp = region[0]
        assert bp.chr1 == bp.chr2
        if bp.bpt1 < bp.bpt2:
            return [EcRegion(bp.chr1, bp.bpt1, bp.bpt2, bp.name, bp.cell)]
        else:
            return [EcRegion(bp.chr1, bp.bpt2, bp.bpt1, bp.name, bp.cell)]
    else:
        new_region = []
        for i in range(len(region)):
            s = region[i]
            e = region[i + 1 - len(region)]
            assert s.chr2 == e.chr1
            new_region.append(EcRegion(s.chr2, s.bpt2, e.bpt1, s.name + e.name, s.cell | e.cell))
        min_bp = sorted(new_region)[0]
        min_idx = new_region.index(min_bp)
        region = new_region[min_idx:] + new_region[:min_idx]
        if region[0].strand == '-':
            for r in region:
                r.reverse()
            region = [region[0]] + region[:0:-1]
        return region

class MergeReads():
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log
        self.maxreadstwoends = self.arg.maxreadstwoends
        self.readsmergeways  = self.arg.readsmergeways
        self.annottype       = self.arg.annottype
        self.overtrimerrors  = self.arg.overtrimerrors
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
        basename = os.path.basename(self.arg.indir)
        #df = pd.read_csv(f'/datd/wangjun/SV/snif/bed/{basename}/oth.bed', sep='\t')
        df = pd.read_csv(f'/share/home/enzedeng/data/ecDNA/summary-SV/SV/bed/{basename}/oth.bed', sep='\t')
        df = df[['#chrom', '#chrom', 'comStart', 'comEnd', 'cell']]
        df.columns = ['chr1', 'chr2', 'bp1', 'bp2', 'cell']
        self.oth = df
        #df = pd.read_csv(f'/datd/wangjun/SV/snif/bed/{basename}/bnd.bed', sep='\t')
        df = pd.read_csv(f'/share/home/enzedeng/data/ecDNA/summary-SV/SV/bed/{basename}/bnd.bed', sep='\t')
        chr_split = df['#chrom'].str.split(';', expand = True)
        loc_split = df['comStart'].str.split(';', expand = True)
        df['chr1'], df['chr2'] = chr_split[0], chr_split[1]
        df['bp1'], df['bp2'] = loc_split[0].astype(int), loc_split[1].astype(int)
        df = df[['chr1', 'chr2', 'bp1', 'bp2', 'cell']]
        self.bnd = df

    def _getinfo(self, _info):
        self.info = _info
        self.inid = _info.sampleid
        self.outdir= '%s/%s'%(self.arg.Merge, self.inid)
        self.arg.outpre= '%s/%s'%(self.outdir, self.inid)
        os.makedirs(self.outdir, exist_ok=True)
        return self

    def _getkeep(self):
        self._inbed= '{0}/{1}/{1}.Keep'.format(self.arg.Search, self.inid)
        if not os.path.exists(self._inbed):
            self.inbed =pd.DataFrame()
            self.log.CW('cannot find the file: ' + self._inbed)
        else:
            self.inbed = pd.read_csv( self._inbed, sep='\t', dtype={'#chrom':str}, low_memory=False)
            self.inbed[['start', 'end']]  = self.inbed[['start', 'end']].astype(int)
            self.inbed['#chrom']  = self.inbed['#chrom'].astype(str)
            #self.inbed['HTSites']  = self.inbed['HTSites'].map(eval)
        return self

    def trimOver(self, _G):
        errors = self.overtrimerrors  #500
        _G['fflag'] = trimOverlink(_G[['#chrom', 'start', 'end', 'forword', 'fflag']].values,
                                    errors=errors)
        return _G

    def orderlinks(self, _G): #check
        _G = _G.reset_index(drop=True)
        _S = _G.sort_values(by= ['length_n', '#chrom', 'start_n', 'end_n'], ascending=[0, 1, 1, 0]).iloc[0].name

        if _G.loc[_S,'forword'] == '+':
            _O = _G.index.tolist()[_S:] +  _G.index.tolist()[:_S]
            _G['forword_n'] = _G['forword']
        else:
            _O = _G.index.tolist()[_S::-1] + _G.index.tolist()[:_S:-1]
            _G['forword_n'] = _G['forword'].replace({'+':'-','-':'+'})

        _G = _G.loc[_O]
        _G['Link'] = _G[['#chrom', 'start_n', 'end_n', 'forword_n']]\
                        .apply(lambda x: '{0}:{1}-{2}'.format(*x[:3]) if x[3] =='+'
                                    else '{0}:{2}-{1}'.format(*x[:3]), axis=1)
        _G['LINKS'] = _G.Link.str.cat(sep=';')
        _G['Order'] = range(1, _G.shape[0] + 1)
        return _G

    def updataLinks(self, inbed):
        sortN = ['SID', 'query_name', 'raw_order']
        gropN = ['SID', 'query_name']
        ColmR = ['#chrom', 'start_n', 'end_n', 'length_n', 'forword', 'raw_order','query_name', 'SID']
        ColmA = ['forword_n', 'Order', 'Link', 'LINKS']

        inbed = inbed.copy()
        inbed['raw_order'] = inbed['raw_order'].astype(int)
        inbed['Type']      = inbed.groupby(by=gropN)['raw_order'].transform(len)
        inbed = inbed.sort_values(by=sortN) #keep raw order right

        #Reduce compution time
        inbed1 = inbed[(inbed.Type <=1)].copy()
        inbed1['forword_n'] = '+'
        inbed1['Order'] = 1
        inbed1['Link'] = inbed1[['#chrom', 'start_n', 'end_n', 'forword']]\
                                    .apply(lambda x: '{0}:{1}-{2}'.format(*x[:3]), axis=1)
        inbed1['LINKS'] = inbed1['Link']

        inbed2 = inbed[(inbed.Type > 1)]
        if inbed2.shape[0] >0:
            outlink = Parallel( n_jobs=-1, backend='loky')( delayed( OrderLinks)(_g[ColmR].to_numpy())
                                for _l, _g in inbed2.groupby(by=gropN, sort=False))
            outlink = pd.DataFrame(np.vstack(outlink), columns=ColmR+ColmA)
            inbed2  = inbed2.merge(outlink, on=ColmR)
        return pd.concat([inbed1, inbed2], axis=0, sort=False)

    def statCircle(self, _G):
        _GO = _G.Order.unique()
        if _GO.size == 1:
            _Q = _G.HTSites.map(InterV2).sum()
            _R = InterVs(_Q)
            _C = InterSm(_R) # reduce time
            _D = InterSm(_Q) # reduce time
            _L = _G.length_n.values[0]
            BPsec, Cover, Depth = _R, _C/_L, _D/_L
        else:
            BPsec = []
            Cover = 1
            Depth = _G.shape[0]

        return pd.Series({
            'LINKS': _G.iloc[0]['LINKS'],
            'LINKSLen' : _G.length_n.sum(),
            'SID'  : _G.iloc[0]['SID'],
            'mapped_reads' : _G.iloc[0]['mapped_reads'],
            'Cover': round(Cover, 3),
            'Depth': round(Depth, 3),
            'BP': BPsec,
            'BPLoc': _GO.tolist(),
            'BPNum': _GO.size,
            'SIDnum' : _G.shape[0],
            'reads' : _G.query_name.str.cat(sep=';')
            })

    def mergeLinks(self, _inbed):
        GRPE  = ['LINKS', 'SID']
        COL1  = ['#chrom', 'start_n', 'end_n', 'length_n', 'Order', 'fflag', 'HTSites', 'query_name', 'mapped_reads']
        #Support = _inbed.loc[(_inbed.fflag.str.contains(';HTBREAKP')), COL1 + GRPE]\
        #            .groupby(by=GRPE, sort=False)\
        #            .apply(lambda x:self.statCircle(x)).reset_index()
        # reduce time
        Support = _inbed.loc[(_inbed.fflag.str.contains(';HTBREAKP')), COL1 + GRPE].groupby(by=GRPE, sort=False)
        Support = Parallel( n_jobs=-1, backend='loky')( delayed( self.statCircle )(_g) for _l, _g in Support)
        Support = pd.concat(Support, axis=1).T.infer_objects()
        Support['EcRPM'] = GroupBY( Support[['SID','SIDnum', 'mapped_reads']].values, _M='EcRPM' )
        #Visal().Lmplot('./aa.pdf', x='SIDnum', y='EcRPM', hue='SID', data=Support)
        return Support

    def stateLinks(self, _inbed, Support):  #slow, need upgrade
        GRPA  = ['LINKS']
        COL2  = ['#chrom', 'start_n', 'end_n', 'Type', 'length_n', 'forword_n', 'LINKS', 'Order']

        inbed = _inbed[COL2].drop_duplicates(keep='first').copy()
        inbed.rename(columns={'start_n': 'start', 'end_n': 'end',
                                'length_n': 'length', 'forword_n': 'forword'}, inplace=True)

        Supgrpb = Support.groupby(by=['LINKS'], sort=True)
        Suplist = [ Supgrpb['SIDnum'].sum().to_frame('Supportsum'),
                    Supgrpb['SID'].size().to_frame('SIDnum'),
                    Supgrpb['Cover'].mean().to_frame('Covermean'),
                    Supgrpb['Depth'].mean().to_frame('Depthmean'),
                    Supgrpb[['BP', 'BPNum']].apply(lambda x: [[]] if x['BPNum'].max() >1 else InterVs(x['BP'].sum())).to_frame('BPs'),
                    Supgrpb['BPLoc'].apply( lambda x: np.unique(x.sum()).tolist()).to_frame('BPLoci'),
                    Supgrpb['BPNum'].max().to_frame('BPNumax'),
                    Supgrpb['BPNum'].max().to_frame('BPLen'),
                    Supgrpb['BPNum'].max().to_frame('LINKSLen'),
                    Supgrpb['SID'].apply(lambda x: x.str.cat(sep=';')).to_frame('SIDs'),
                    Supgrpb['SIDnum'].apply(lambda x: x.astype(str).str.cat(sep=';')).to_frame('Supports'),
                    Supgrpb['Cover'].apply(lambda x: x.astype(str).str.cat(sep=';')).to_frame('Covers'),
                    Supgrpb['Depth'].apply(lambda x: x.astype(str).str.cat(sep=';')).to_frame('Depths'),
                    Supgrpb['BPNum'].apply(lambda x: x.astype(str).str.cat(sep=';')).to_frame('BPNums') ]
        Suplist = pd.concat( Suplist, ignore_index=False, join = 'outer', sort = False, axis=1).reset_index()
        del Supgrpb

        inbed = inbed.merge(Suplist, on='LINKS', how='outer')
        del Suplist

        inbed['BPLoci'] = inbed.apply(lambda x: 'BP' if x.Order in x.BPLoci else '', axis=1)
        inbed['BPs']    = inbed.apply(lambda x:  x.BPs if (x.BPLoci =='BP' and x.BPNumax ==1) else [[x.start, x.end]], axis=1)
        inbed['BPLen']  = inbed['BPs'].map(InterSm)
        inbed['LINKSLen']  = inbed.groupby(GRPA)['BPLen'].transform('sum')
        return inbed

    def mergeSample(self, _inbed):
        GRPBY = ['SID', 'query_name']
        bp_list = Parallel( n_jobs= -1, backend='threading')( delayed( self.convert_to_bps )(_g)
                   for _, _g in _inbed.groupby(by=GRPBY, sort=False))
        bp_list = [item for sublist in bp_list for item in sublist]
        bp_list = self.mergeBp(bp_list)
        region_list = self.mergeRegion(bp_list)
        self.statRegion(region_list, _inbed)

    def mergeMultiSample(self, _inbed):
        GRPBY = ['SID', 'query_name']
        bp_list = Parallel( n_jobs= -1, backend='threading')( delayed( self.convert_to_bps )(_g)
                   for _, _g in _inbed.groupby(by=GRPBY, sort=False))
        bp_list = [item for sublist in bp_list for item in sublist]
        bp_list = self.mergeBp(bp_list)
        region_list = self.mergeRegion(bp_list)
        self.statSingleRegion(region_list, _inbed)

    def statRegion(self, region_list, _inbed):
        with open(self.arg.outpre+'.stat', 'w') as f:
            for region in sorted(region_list):
                f.write('%s\n' % '\t'.join(x.serialize_to_string() for x in region))
        info_list = Parallel( n_jobs= -1, backend='threading')( delayed( self.stat_region )(_r, _inbed) for _r in sorted(region_list))
        indf = pd.concat(info_list, axis=1, sort=False).transpose()
        indf.sort_values(by=['Type', '#chrom', 'start', 'end'], inplace=True)
        indf.to_csv(self.arg.outpre+'.UpMerge', sep='\t', index=False)

    def stat_region(self, region, _inbed):
        total_length = 0
        total_base = 0
        total_cover = 0
        read_num = 0
        query_num = 0
        cnv = 0
        support_num = []
        new_region = []
        length_list = []
        link_length_list = []
        cover_list = []
        depth_list = []
        names = []
        types = []
        sid = set()
        sid_dict = collections.defaultdict(int)
        for r in region:
            new_region.append(EcRegion(r.chr, r.start, r.end, [], r.cell))
            sid.update(r.cell)
            length = abs(r.end - r.start) + 1
            query_name = r.name
            query = _inbed[ (_inbed['#chrom']==r.chr) &
                (_inbed['SID'].isin(r.cell)) &
                (_inbed['query_name'].isin(query_name)) &
                (_inbed['start'] >= r.start - self.maxreadstwoends) &
                (_inbed['end'] <= r.end + self.maxreadstwoends) ]
            cover_base = np.zeros(length)
            for i, q in query.iterrows():
                query_num += 1
                cnv += q['cnv']
                sid_dict[q['SID']] += 1
                s = q['start'] - r.start
                e = q['end'] - r.start
                s = 0 if s < 0 else s
                cover_base[s:e] += 1
            read_num += len(r.name)
            support_num.append(len(r.name))
            total_length += length
            length_list.append(length)
            total_base += np.sum(cover_base)
            total_cover += np.sum(cover_base!=0)
            link_length_list.append(np.sum(cover_base!=0))
            cover_list.append(np.sum(cover_base!=0) / length)
            depth_list.append(np.sum(cover_base) / length)
            name, type = Annotate(self.arg, self.log).geneannote(r.chr, r.start, r.end)
            names.append(name)
            types.append(type)
        #print(region, read_num, support_num, total_length, total_base, total_cover, total_base/total_length, total_cover/total_length, cnv/read_num)
        if not query_num:
            return
        return pd.concat([pd.Series({
            '#chrom': region.chr,
            'start': region.start,
            'end': region.end,
            'Type': 2 if len(new_region) > 1 else 1,
            'TotalLength': total_length,
            'forword': region.strand,
            'Region': ','.join(str(x) for x in new_region),
            'RegionNum': len(new_region),
            'Length': ','.join(str(x) for x in length_list),
            'ReadNum' : read_num / 2 if len(new_region) > 1 else read_num,
            'SupportNum': ','.join(str(x) for x in support_num),
            'SIDNum': len(sid),
            'SID' : ','.join(sorted(sid)),
            'SIDNums'  : ','.join([f'{int(sid_dict[x]/2)}' for x in sorted(sid)]),
            'LinkLength': sum(link_length_list),
            'LinkLengths': ','.join('%d' % x for x in link_length_list),
            'Cover': round(total_cover / total_length, 3),
            'Covers': ','.join('%.3f' % x for x in cover_list),
            'Depth': round(total_base / total_length, 3),
            'Depths': ','.join('%.3f' % x for x in depth_list),
            'CNV' : round(cnv / read_num / 2, 3),
            'gene_name': names[idx],
            'gene_biotype': types[idx],
            }) for idx, region in enumerate(new_region)], axis=1, sort=False)

    def checkSV(self, bp, dist=500):
        # TODO: should add to log
        if bp.chr1 == bp.chr2:
            df = self.oth
            df = df[df['chr1'] == bp.chr1]
            df = df[((abs(df['bp1'] - bp.bpt1) < dist) & (abs(df['bp2'] - bp.bpt2) < dist)) |
                    ((abs(df['bp1'] - bp.bpt2) < dist) & (abs(df['bp2'] - bp.bpt1) < dist))]
            for index, row in df.iterrows():
                for c in row['cell'].split(';'):
                    if c in bp.cell:
                         return True
        else:
            df = self.bnd
            df1 = df[(df['chr1'] == bp.chr1) & (df['chr2'] == bp.chr2)]
            df2 = df[(df['chr1'] == bp.chr2) & (df['chr2'] == bp.chr1)]
            df1 = df1[(abs(df1['bp1'] - bp.bpt1) < dist) & (abs(df1['bp2'] - bp.bpt2) < dist)]
            df2 = df2[(abs(df2['bp1'] - bp.bpt2) < dist) & (abs(df2['bp2'] - bp.bpt1) < dist)]
            for index, row in df1.iterrows():
                for c in row['cell'].split(';'):
                    if c in bp.cell:
                         return True
            for index, row in df2.iterrows():
                for c in row['cell'].split(';'):
                    if c in bp.cell:
                         return True
        return False


    def mergeBp(self, bp_list):
        sorted_bp_list = sorted(bp_list) # , key=lambda x: (x.chr1, x.chr2, x.bpt1, x.bpt2, x.dir1, x.dir2))
        merged_bp_dict = collections.defaultdict(lambda: collections.defaultdict(list))
        with open(self.arg.outpre+'.bp', 'w') as f:
            for bp in sorted_bp_list:
                f.write(f'{bp}\n')
                merged_bp_dict[bp.chr1][bp.chr2].append(bp)
        merged_list = []
        for chr1 in merged_bp_dict:
            for chr2 in merged_bp_dict[chr1]:
                #print(chr1, chr2, len(merged_bp_dict[chr1][chr2]))
                merged_bp_list = [merged_bp_dict[chr1][chr2][0]]
                for i in range(1, len(merged_bp_dict[chr1][chr2])):
                    new = merged_bp_dict[chr1][chr2][i]
                    merged = False
                    for old in merged_bp_list[-1:-42:-1]:
                        if (abs(new.bpt1 - old.bpt1) < self.maxreadstwoends and
                            abs(new.bpt2 - old.bpt2) < self.maxreadstwoends and
                            new.dir1 == old.dir1 and new.dir2 == old.dir2):
                            old.name.append(new.name[-1])
                            old.cell.update(new.cell)
                            if old.dir1 == '+':
                                old.bpt1 = min(old.bpt1, new.bpt1)
                            else:
                                old.bpt1 = max(old.bpt1, new.bpt1)
                            if old.dir2 == '+':
                                old.bpt2 = min(old.bpt2, new.bpt2)
                            else:
                                old.bpt2 = max(old.bpt2, new.bpt2)
                            merged = True
                            break
                    if not merged:
                        merged_bp_list.append(new)
                merged_list.extend(merged_bp_list)
        merged_bp_dict = collections.defaultdict(lambda: collections.defaultdict(list))
        with open(self.arg.outpre+'.bpFilter', 'w') as f:
            for bp in merged_list:
                # TODO(add flag)
                #if bp.chr1 != bp.chr2:
                #    continue
                if bp.chr1 not in self.chrs or bp.chr2 not in self.chrs:
                    f.write(f"CHR_FILTER\t{bp}\n")
                    continue
                if abs(bp.bpt1 - bp.bpt2) < self.maxreadstwoends:
                    f.write(f"LENGTH_FILTER\t{bp}\n")
                    continue
                if len(bp.name) < 3: # TODO support reads less than 3
                    f.write(f"SUPPORT_FILTER\t{bp}\n")
                    continue
                if self.checkSV(bp):
                    f.write(f"SV_FILTER\t{bp}\n")
                    continue
                merged_bp_dict[bp.chr1][bp.chr2].append(bp)
        # merge again
        clean_bp_list = []
        for chr1 in merged_bp_dict:
            for chr2 in merged_bp_dict[chr1]:
                merged_bp_list = [merged_bp_dict[chr1][chr2][0]]
                for i in range(1, len(merged_bp_dict[chr1][chr2])):
                    new = merged_bp_dict[chr1][chr2][i]
                    merged = False
                    for old in merged_bp_list[-1:-42:-1]:
                        if (abs(new.bpt1 - old.bpt1) < self.maxreadstwoends and
                            abs(new.bpt2 - old.bpt2) < self.maxreadstwoends and
                            new.dir1 == old.dir1 and new.dir2 == old.dir2):
                            old.name.append(new.name[-1])
                            old.cell.update(new.cell)
                            if old.dir1 == '+':
                                old.bpt1 = min(old.bpt1, new.bpt1)
                            else:
                                old.bpt1 = max(old.bpt1, new.bpt1)
                            if old.dir2 == '+':
                                old.bpt2 = min(old.bpt2, new.bpt2)
                            else:
                                old.bpt2 = max(old.bpt2, new.bpt2)
                            merged = True
                            break
                    if not merged:
                        merged_bp_list.append(new)
                clean_bp_list.extend(merged_bp_list)
        with open(self.arg.outpre+'.bpMerge', 'w') as f:
            [f.write(f'{bp}\n') for bp in sorted(clean_bp_list)]
        return clean_bp_list

    def findRegion(self, bp, same=False):
        bp0 = bp
        bps = [bp0]
        itertimes = 0
        while True:
            bp = self.find_nearest_breakpoint(bp, same=same)
            if not bp:
                bps = []
                break
            if bp in bps:
                bps = bps[bps.index(bp):]
                break
            bps.append(bp)
            itertimes += 1
            if itertimes > 42:
                bps = []
                break
        if bps:
            return trans_region(bps)

    def selfCircle(self, bp):
        if bp.chr1 != bp.chr2:
            return
        if ((bp.bpt1 < bp.bpt2 and bp.dir1 == '+' and bp.dir2 == '-') or
            (bp.bpt2 < bp.bpt1 and bp.dir2 == '+' and bp.dir1 == '-')):
            return trans_region([bp])

    def mergeRegion(self, bp_list):
        self.region_dict = collections.defaultdict(lambda: collections.defaultdict(list))
        for bp in bp_list:
            chr1 = bp.chr1
            chr2 = bp.chr2
            self.region_dict[chr1][chr2].append(bp)
        region_list = []
        idx = 0
        for bp in bp_list:
            idx += 1
            if idx % 1000 == 0:
                print(idx, "breakpoint processed")
            region = self.selfCircle(bp)
            if region and region not in region_list:
                region_list.append(region)
            region = self.findRegion(bp)
            if region and region not in region_list:
                region_list.append(region)
            region = self.findRegion(bp, same=True)
            if region and region not in region_list:
                region_list.append(region)
            region = self.findRegion(bp.reverse())
            if region and region not in region_list:
                region_list.append(region)
            region = self.findRegion(bp.reverse(), same=True)
            if region and region not in region_list:
                region_list.append(region)
        return sorted(region_list)

    def find_nearest_breakpoint(self, bp0, mind=90, same=False):
        if not bp0:
            return None
        bp_dict = self.region_dict
        if same:
            min_dist = 1e9
            bpns = None
            for bp in bp_dict[bp0.chr2][bp0.chr2]:
                assert bp.chr1 == bp0.chr2
                if bp.cell != bp0.cell:
                    continue
                if bp0.dir2 == '+' and bp.dir1 == '-':
                    dist = bp.bpt1 - bp0.bpt2
                    if mind < dist < min_dist:
                        bpns = bp
                        min_dist = dist
                elif bp0.dir2 == '-' and bp.dir1 == '+':
                    dist = bp0.bpt2 - bp.bpt1
                    if mind < dist < min_dist:
                        bpns = bp
                        min_dist = dist
            if not bpns:
                return None
            return BreakPoint(bpns.chr1, bpns.bpt1, bpns.dir1, bpns.chr2, bpns.bpt2, bpns.dir2, bpns.name, bpns.cell, sort=False)
        min_dist = 1e9
        bpn1, bpn2 = None, None
        for chr2 in bp_dict[bp0.chr2]:
            for bp in bp_dict[bp0.chr2][chr2]:
                assert bp.chr1 == bp0.chr2
                if bp.cell != bp0.cell:
                    continue
                if bp0.dir2 == '+' and bp.dir1 == '-':
                    dist = bp.bpt1 - bp0.bpt2
                    if mind < dist < min_dist:
                        bpn1 = bp
                        min_dist = dist
                elif bp0.dir2 == '-' and bp.dir1 == '+':
                    dist = bp0.bpt2 - bp.bpt1
                    if mind < dist < min_dist:
                        bpn1 = bp
                        min_dist = dist
        min_dist = 1e9
        for chr1 in bp_dict:
            for bp in bp_dict[chr1][bp0.chr2]:
                if bp.cell != bp0.cell:
                    continue
                if bp0.dir2 == '+' and bp.dir2 == '-':
                    dist = bp.bpt2 - bp0.bpt2
                    if mind < dist < min_dist:
                        bpn2 = bp
                        min_dist = dist
                elif bp0.dir2 == '-' and bp.dir2 == '+':
                    dist = bp0.bpt2 - bp.bpt2
                    if mind < dist < min_dist:
                        bpn2 = bp
                        min_dist = dist
        if not bpn1 and not bpn2:
            return None
        if not bpn1:
            return BreakPoint(bpn2.chr2, bpn2.bpt2, bpn2.dir2, bpn2.chr1, bpn2.bpt1, bpn2.dir1, bpn2.name, bpn2.cell, sort=False)
        if not bpn2:
            return BreakPoint(bpn1.chr1, bpn1.bpt1, bpn1.dir1, bpn1.chr2, bpn1.bpt2, bpn1.dir2, bpn1.name, bpn1.cell, sort=False)
        assert bpn1.chr1 == bp0.chr2 and bpn2.chr2 == bp0.chr2
        if abs(bpn1.bpt1 - bp0.bpt2) < abs(bpn2.bpt2 - bp0.bpt2):
            return BreakPoint(bpn1.chr1, bpn1.bpt1, bpn1.dir1, bpn1.chr2, bpn1.bpt2, bpn1.dir2, bpn1.name, bpn1.cell, sort=False)
        else:
            return BreakPoint(bpn2.chr2, bpn2.bpt2, bpn2.dir2, bpn2.chr1, bpn2.bpt1, bpn2.dir1, bpn2.name, bpn2.cell, sort=False)

    def convert_to_bps(self, df):
        if len(df.index) < 2:
            return []
        bp_list = []
        for i in range(len(df.index) - 1):
            f = df.loc[df.index[i]]
            s = df.loc[df.index[i+1]]
            if f['cigarreg'][0] > s['cigarreg'][0]:
                f, s = s, f
            if f['forword'] == '+':
                bpt1 = f['end']
                dir1 = '-'
            else:
                bpt1 = f['start']
                dir1 = '+'
            if s['forword'] == '+':
                bpt2 = s['start']
                dir2 = '+'
            else:
                bpt2 = s['end']
                dir2 = '-'
            if f['#chrom'] == s['#chrom'] and dir1 == dir2:
                continue
            bp = BreakPoint(f['#chrom'], bpt1, dir1,
                            s['#chrom'], bpt2, dir2, f['query_name'], f['SID'])
            bp_list.append(bp)
        return bp_list

    def mergeReads(self, _inbed, Lplot=True, Hplot=False):
        self.mergeSample(_inbed)
        return
        if len(np.unique(_inbed['SID'])) == 1:
            pass
        else:
            self.mergeMultiSample(_inbed)
        return

        # Trim duplicated link
        GRPBY = ['SID', 'query_name']
        inbed = Parallel( n_jobs= -1, backend='threading')( delayed( self.trimOver )(_g)
                   for _, _g in _inbed.groupby(by=GRPBY, sort=False))
        inbed = pd.concat(inbed, axis=0, sort=False)
        inbed.to_csv(self.arg.outpre+'.Keep', sep='\t', index=False)
        inbed = inbed[~(inbed.fflag.str.contains('Trim|MultiChr', regex=True))]

        # mergetwoends
        inbed = Utilities(self.arg, self.log)\
                    .mapanytwo(inbed, maxdistance = self.maxreadstwoends, maxreg = self.readsmergeways)

        # merge breakpoint
        inbed = MergeReads(self.arg, self.log).updataLinks(inbed)
        inbed.to_csv(self.arg.outpre+'.Links', sep='\t', index=False)

        # merge links
        Support = MergeReads(self.arg, self.log).mergeLinks(inbed)
        Support.to_csv(self.arg.outpre+'.Support', sep='\t', index=False)

        # state links
        inbed   = MergeReads(self.arg, self.log).stateLinks(inbed, Support)
        inbed.to_csv(self.arg.outpre+'.LinksUp', sep='\t', index=False)

        if self.annottype =='bp':
            inbed = Annotate(self.arg, self.log).geneannotb(inbed)
        elif self.annottype =='all':
            inbed = Annotate(self.arg, self.log).geneannota(inbed)
        elif self.annottype =='part':
            inbed = Annotate(self.arg, self.log).geneannotc(inbed)

        inbed['BPs'] = inbed.apply(lambda x: ';'.join(['%s:%s-%s'%(x['#chrom'],i[0], i[1]) for i in x['BPs']]), axis=1)
        inbed.to_csv(self.arg.outpre+'.UpMerge', sep='\t', index=False)

        inbed = inbed.sort_values(by=['Type', 'Order', '#chrom', 'start', 'end', 'LINKS'])
        inbed.to_csv(self.arg.outpre+'.UpMerge_sort', sep='\t', index=False)

        if Lplot:
            Visal().query_length(inbed, self.arg.outpre+'.UpMerge.BPlength.pdf',
                                    X='BPLen', Dup='', log=True, title='breakpoint length')
            Visal().query_length(inbed, self.arg.outpre+'.UpMerge.length.pdf',
                                    X='LINKSLen', Dup='LINKS', log=True, title='Links length')
            Visal().Pdistribut( inbed,  self.arg.outpre+'.supportnum2-30.pdf',
                                    X='Supportsum', Dup=['LINKS'], logx=False, logy=True,
                                    Bins = range(1,31), xlim = [0, 30], title='support number')
            #Visal().clustmap(pvot, self.outpre+'.Keep.matrix.pdf')
            #Visal().clustmap(np.log2(pvot+1), self.outpre+'.Keep.matrix.log2.pdf')

    def EachEcDNA(self, _info):
        self._getinfo(_info)
        self._getkeep()
        self.log.CI('start merging breakpoin region: ' + self.inid)
        if not self.inbed.empty:
            self.mergeReads( self.inbed )
        else:
            self.log.CW('cannot find any circle region singal: ' + self.inid)
        self.log.CI('finish merging breakpoin region: ' + self.inid)
