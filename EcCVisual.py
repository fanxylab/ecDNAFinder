#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
***********************************************************
* @File    : get_links.py
* @Author  : Zhou Wei                                     *
* @Date    : 2021/01/25 19:41:11                          *
* @E-mail  : welljoea@gmail.com                           *
* @Version : --                                           *
* You are using the program scripted by Zhou Wei.         *
* If you find some bugs, please                           *
* Please let me know and acknowledge in your publication. *
* Thank you!                                              *
* Best wishes!                                            *
***********************************************************
'''

import pandas as pd
import numpy  as np
import re
import os
#import pyranges as pr

class Circos:
    def __init__(self, arg, log):
        self.arg  = arg
        self.log  = log

    def _getio(self):
        self.circonf = os.path.dirname(os.path.realpath(__file__)) + '/../Circos/links.circos.conf'
        self.infile  = (self.arg.circisin  or '%s/%s.UpFilterTRF'%(self.arg.Update, self.arg.updatepre) )
        self.outdir  = (self.arg.circisout or self.arg.Update)
        self.genefl  = self.arg.genefilt
        self.keepbio = self.arg.keepbio
        self.outdata = self.outdir + '/data/'
        self.outhead = self.arg.cirhead
        self.outpre  = "%s/%s"%(self.outdir, self.outhead)
        self.Chrs    = [ 'hs' + str(i) for i in range(1,23)] + ['hsX','hsY']
        os.makedirs(self.outdata, exist_ok=True)

    def _getinfo(self):
        Links = pd.read_csv(self.infile, sep='\t')
        Links['#chrom'] = 'hs' + Links['#chrom'].astype(str).str.lstrip('chr')
        self.Links =  Links[(Links['#chrom'].isin(self.Chrs))]
        self.Links.to_csv('test.txt', sep='\t')
        #Drop  = Links.loc[ ~(Links['#chrom'].isin(self.Chrs)), 'Region' ]
        #self.Links = Links[ ~(Links.Region.isin(Drop)) ]

        Gene = pd.read_csv(self.arg.gtfbed, sep='\t')
        Gene['#chrom'] = 'hs' + Gene['#chrom'].astype(str).str.lstrip('chr')
        self.Gene = Gene.loc[(Gene.gene_biotype.isin(self.keepbio)), ['#chrom', 'start', 'end', 'gene_name']]
        if self.genefl:
            cgene = pd.read_csv(self.genefl, sep='\t').iloc[:,0]
            self.Gene = self.Gene[(self.Gene.gene_name.isin(cgene))]

    def linkstack(self, _G):
        '''
        '#chrom', 'start', 'end', 'ReadNum'
        '''
        if _G.shape[0] <2:
            return _G
        _G = _G.copy()
        _G[:, 1:3]  = np.sort(_G[:, 1:3]) #_G[:,[1:3]].sort()
        _L = sorted(set(_G[:,1:3].flatten()))
        _L = np.array([ [_G[0,0], _L[i], _L[i+1], 0] for i in range(len(_L) -1 )], dtype=np.object)
        for _l in _G:
            _L[ (_L[:,1]>=_l[1]) & (_L[:,2]<=_l[2]), 3] += _l[3]
        return _L[ (_L[:,3]>0), :]

    def bpstack(self, _G):
        '#chrom', 'start', 'end', 'strand', 'ReadNum'
        _B = np.sort( np.unique( _G[:,1:3].flatten()) )
        _L = _B.shape[0] - 1

        _B = np.c_[ np.repeat(_G[0,0],_L), _B[:-1], _B[1:], np.zeros(_L, dtype=int) ]
        for _g in _G:
            _B[(_B[:,1]>=_g[1]) & (_B[:,2] <=_g[2]), 3] += _g[4]
        return _B[(_B[:,3]>0)]

    def linktrans(self, _L):
        _l = _L.iloc[0]
        _B = []
        for i in _l.Region.split(','):
            k = re.split('[-:]',i)
            if len(k) != 4:
                return np.array([], dtype=np.int64).reshape(0,7)
            for j in k[1:3]:
                _B.append( [ 'hs'+k[0], int(j), int(j)+1 ])
        _C = [ _B[i] + _B[i+1] + ['type=link'] for i in range(len(_B)-1) ]

        if _l.Type > 1:
            _C.append( _B[-1] + _B[0] + ['type=link'] )

        _p = re.split('[\-:,]', _l.Region)
        if len(_p) > 4:
            _C.append([ 'hs'+_p[0], int(_p[2]), int(_p[2])+1,
                        'hs'+_p[4], int(_p[5]), int(_p[5])+1, 'type=headtail'])
        else:
            _C.append([ 'hs'+_p[0], int(_p[1]), int(_p[1])+1,
                        'hs'+_p[0], int(_p[2]), int(_p[2])+1, 'type=circos'])
        return _C

    def linksnum(self, Links):
        Links = Links.copy()
        targetdf = Links[['#chrom', 'start', 'end', 'ReadNum']]
        K=targetdf.groupby('#chrom').apply(lambda x: self.linkstack(x.values) )
        np.savetxt( self.outdata +'/links.num.txt', np.vstack(K), delimiter='\t',  fmt='%s')

    def bpsnum(self, Links):
        T = [ re.split('[:-]', _j) + [_i[1]] for _i in  Links[['Region', 'ReadNum']].values for _j in _i[0].split(',') ]
        T = [x for x in T if len(x) == 5]
        T = [[x[0], int(x[1]), int(x[2]), x[3], int(x[-1])] for x in T]
        T = [[x[0], x[1]-5000, x[1]+5000, x[3], x[-1]] for x in T] + [[x[0], x[2]-5000, x[2]+5000, x[3], x[-1]] for x in T]
        T = pd.DataFrame(T, columns=['#chrom', 'start', 'end', 'strand', 'ReadNum'])
        T['#chrom'] = 'hs' + T['#chrom'].astype(str).str.lstrip('chr')
        T[['start', 'end', 'ReadNum']] = T[['start', 'end', 'ReadNum']].astype(int)
        K = T.groupby('#chrom').apply(lambda x: self.bpstack(x.values) )
        np.savetxt( self.outdata +'/links.num.txt', np.vstack(K), delimiter='\t',  fmt='%s')

    def mulitlinks(self, Links):
        Linkmelt = []
        for i in Links.LINKS.unique():
            fillinks = [re.split('[:-]',i) for i in i.split(';')]
            if len(fillinks)>1:
                for j in range(len(fillinks) -1 ):
                    Linkmelt.append( fillinks[j] + fillinks[j+1] + ['type=%s'%len(fillinks)] )

        Linkmelt = pd.DataFrame(Linkmelt, columns=['chra','sa', 'ea', 'chrb', 'sb', 'eb', 'ty'])
        Linkmelt.sort_values(by=['chra','sa', 'ea'],inplace=True)
        Linkmelt.iloc[:,0] = 'hs' + Linkmelt.iloc[:,0].astype(str)
        Linkmelt.iloc[:,3] = 'hs' + Linkmelt.iloc[:,3].astype(str)
        #Linkmelt.iloc[:,1:3].values.sort()
        #Linkmelt.iloc[:,4:6].values.sort()
        Linkmelt.to_csv(self.outdata + '/links.multi.txt', sep='\t',index=False, header=False)

    def linkssite(self, Links):
        targetdf = Links[['#chrom', 'start', 'end', 'Type']].copy()
        targetdf['Type'] = 'type=' + targetdf['Type'].astype(str)
        targetdf.to_csv( self.outdata + '/links.site.txt', sep='\t',index=False, header=False)

    def linksbps(self, Links):
        K = Links.groupby('Region').apply(lambda x: self.linktrans(x) )
        np.savetxt( self.outdata +'/links.txt', np.vstack(K), delimiter='\t', fmt='%s')

    def geneannot(self, Links, Gene):
        targetseri = list(set(';'.join(Links.gene_name.fillna('.')).split(','))) # ; to ,
        Gene[ (Gene.gene_name.isin(targetseri))]\
            .to_csv( self.outdata +'/links.gene.txt', sep='\t',index=False, header=False)

    def goplot(self):
        cmd='cd {2} ; {4} {0} -conf {1} -outputdir {2} -outputfile {3}.svg'\
                .format(self.arg.circissw, self.circonf, self.outdir, self.outhead, self.arg.perl )

        if self.arg.cirplot:
            os.system(cmd)
            try:
                import cairosvg
                cairosvg.svg2pdf(url=self.outpre+'.svg', write_to=self.outpre+'.pdf')
                #from svglib.svglib import svg2rlg
                #from reportlab.graphics import renderPDF
                #drawing = svg2rlg("%s/%s.svg"%(self.outdir, self.outhead) )
                #renderPDF.drawToFile(drawing, "%s/%s.pdf"%(self.outdir, self.outhead))
                #os.system('rsvg-convert {0}/{1}.svg -f pdf -o {0}/{1}.1.pdf'.format(self.outdir, self.outhead))
            except:
                pass

    def ConfData(self):
        self.log.CI('start circos plot.')
        self._getio()
        self._getinfo()
        #self.linksnum(self.Links.copy())
        #self.mulitlinks(self.Links.copy())
        #self.linkssite(self.Links.copy())
        self.bpsnum(self.Links.copy())
        self.linksbps(self.Links.copy())
        self.geneannot(self.Links.copy(), self.Gene.copy())
        #self.goplot()
        self.log.CI('finish circos plot.')
