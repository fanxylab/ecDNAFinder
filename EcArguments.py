#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

def Args():
    parser = argparse.ArgumentParser(
                formatter_class=argparse.RawDescriptionHelpFormatter,
                prefix_chars='-+',
                conflict_handler='resolve',
                description="",
                epilog="")

    parser.add_argument('-V','--version',action ='version',
                version='EcDNA version 0.1')

    subparsers = parser.add_subparsers(dest="commands",
                    help='models help.')
    P_Common = subparsers.add_parser('Common',conflict_handler='resolve', #add_help=False,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    help='The common parameters used for other models.')
    P_Common.add_argument("-f", "--infile",type=str,
                    help='''the input file or input number split by ",".''')
    P_Common.add_argument("-i", "--indir",type=str,
                    help='''the input directory.''')
    P_Common.add_argument("-o", "--outdir",type=str, default=os.getcwd(),
                    help="output file dir. [Default: %(default)s]")
    P_Common.add_argument("-n", "--njob",type=int, default=5,
                    help="The maximum number of concurrently running jobs. [Default: %(default)s]")
    P_Common.add_argument("-bd", "--bamdir", type=str, default='02.MiniMap',
                    help="input bam directory for fetch module. [Default: %(default)s]")
    P_Common.add_argument("-fd", "--fetchdir", type=str,  default='03.SoftMap',
                    help="out directory for fetch. [Default: %(default)s]")
    P_Common.add_argument("-sd", "--searchdir", type=str, default='03.SoftMap',
                    help="out directory of search. [Default: %(default)s]")
    P_Common.add_argument("-md", "--mergedir", type=str,  default='03.SoftMap',
                    help="out directory of merge. [Default: %(default)s]")
    P_Common.add_argument("-vd", "--cnvdir", type=str,  default='04.CNV',
                    help="out directory for CNV. [Default: %(default)s]")
    P_Common.add_argument("-ud", "--updatedir", type=str, default='04.EcRegion',
                    help="out directory for update. [Default: %(default)s]")
    P_Common.add_argument("-cd", "--checkdir", type=str, default='05.CheakBP',
                    help="out directory for check breakpoint of  plasmid. [Default: %(default)s]")
    P_Common.add_argument("-bt", "--bedtools", type=str, default='/share/home/share/software/bedtools2/bin/',
                    help="bedtools path. [Default: %(default)s]")
    P_Common.add_argument("-st", "--samtools", type=str, default='/share/home/share/software/samtools-1.10/bin/',
                    help="samtools path. [Default: %(default)s]")
    P_Common.add_argument("-gt", "--gtf", type=str,
                    default='/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf',
                    help="the genome gtf file. [Default: %(default)s]")

    P_cnv = subparsers.add_parser('CNV', conflict_handler='resolve', add_help=False)
    P_cnv.add_argument("-ng", "--ngaps", type=str,
                    default='/share/home/share/Repository/GenomeDB/Blacklist/hg38_ucsc_track_gap.txt',
                    help="the hg38 ucsc track N gaps.")
    P_cnv.add_argument("-gm", "--genomecnv", type=str,
                    default='/share/home/zhou_wei/Workspace/01Repository/GenomeDB/Reference/EcDNARef/HG38_ENSEMBL_Plasmid20.fa',
                    help="the genome fasta.")
    P_cnv.add_argument("-ct", "--cytoband", type=str,
                    default='/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/UCSC/HG38/hg38_ucsc_cytoBand.txt',
                    help="the cytoband file.")
    P_cnv.add_argument("-bl", "--blacklist", type=str,
                    default='/share/home/share/Repository/GenomeDB/Blacklist/hg38_ENCFF356LFX_unified_blacklist.bed',
                    help="the bed file hg38 ENCFF356LFX unified blacklist.")
    P_cnv.add_argument("-bi", "--buildidx", type=str,
                    help="the index file built of reference cnv bin. If not specified, it will find or build in the data path.")
    P_cnv.add_argument("-sb", "--splitbin", type=int, default=5000,
                    help="the legnth of each split bin.")
    P_cnv.add_argument("-db", "--dropbinlen", type=int, default=1000,
                    help="the min legnth of each Ngaps region.")
    P_cnv.add_argument("-ed", "--endindepfre", type=float, default=0.65,
                    help="the min ratio of last split bin lenght.")
    P_cnv.add_argument("-mb", "--mergebin", type=int, default=500000,
                    help="the legnth of ecah merge bin.")
    P_cnv.add_argument("-eb", "--endmergepfre", type=float, default=0.45,
                    help="the min ratio of last merge bin lenght. [Default: %(default)s]")
    P_CNV = subparsers.add_parser('CNV',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_cnv],
                    help='get CNV information.')

    P_fetch = subparsers.add_parser('Fetch', conflict_handler='resolve', add_help=False)
    P_fetch.add_argument("-ms", "--minsoftdrop",  type=int, default=20,
                    help="the min length of softclip to drop. [Default: %(default)s]")
    P_fetch.add_argument("-mq", "--minmapQ", type=int, default=0,
                    help="the min mapq of align reads. [Default: %(default)s]")
    P_fetch.add_argument("-gs", "--getsoftfq", action='store_true', default=False,
                    help="whether to get softclip reads with fastq format. [Default: %(default)s]")
    P_fetch.add_argument("-sl", "--lensoftfq",  type=int, default=100,
                    help="the minimun softclip length to save. [Default: %(default)s]")
    P_fetch.add_argument("-mi", "--maskindel", type=int, default=100000,
                    help="the number to mask indel in cigar tulpe. [Default: %(default)s]")
    P_fetch.add_argument("-ms", "--maskskip", type=int, default=10000000,
                    help="the number to mask skip in cigar tulpe. [Default: %(default)s]")
    P_fetch.add_argument("-mh", "--maskhard", type=int, default=100000,
                    help="the number to mask hard softclip in cigar tulpe. [Default: %(default)s]")
    P_fetch.add_argument("-mp", "--maskpad", type=int, default=10000000,
                    help="the number to mask pad in cigar tulpe. [Default: %(default)s]")
    P_Fetch = subparsers.add_parser('Fetch',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common,P_fetch],
                    help='fatch reads information from bam file.')

    P_search = subparsers.add_parser('Search', conflict_handler='resolve', add_help=False)
    P_search.add_argument("-dc", "--dropcigarover", action='store_true', default=True,
                    help="whether to drop the overlap mapping region. [Default: %(default)s]")
    P_search.add_argument("-dc", "--dropneighbdup", action='store_true', default=True,
                    help="whether to drop the duplication of nerghbor mapping region. [Default: %(default)s]")
    P_search.add_argument("-oe", "--overmaperrors", type=int, default=100,
                    help="the error margion bases in overlap mapping region. [Default: %(default)s]")
    P_search.add_argument("-na", "--minalignlenght", type=int, default=100,
                    help="the minimum lenght of alignment. [Default: %(default)s]")
    P_search.add_argument("-nl", "--minbplenght", type=int, default=300,
                    help="the minimum lenght of breakpoint. [Default: %(default)s]")
    P_search.add_argument("-xl", "--maxbplenght", type=int, default=1000000000,
                    help="the maximum lenght of breakpoint. [Default: %(default)s]")
    P_search.add_argument("-ht", "--maxhtdistance", type=int, default=10000000,
                    help="if the distance of breakpoint is large than the number, the warning work out. [Default: %(default)s]")
    P_search.add_argument("-nt", "--maxneighbtwoends", type=int, default=250,
                    help="the max distance of breakpoint of two ends to merge nerghbour mapping region. [Default: %(default)s]")
    P_search.add_argument("-no", "--maxneighboneend", type=int, default=100,
                    help="the max distance of breakpoint of one end to merge nerghbour mapping region. [Default: %(default)s]")
    P_search.add_argument("-bl", "--minneighbplen", type=int, default=500,
                    help="the min lenght of breakpoint of two nerghbour reads. [Default: %(default)s]")
    P_search.add_argument("-gl", "--mingap", type=int, default=100,
                    help="the min lenght of gap of two nerghbour reads. [Default: %(default)s]")
    P_search.add_argument("-nw", "--neighbmergeways", action='store_true', default=True,
                    help="whether to use the max distance of breakpoint to merge nerghbour mapping region. [Default: %(default)s]")
    P_search.add_argument("-nm", "--maxmasksofttwoends", type=float, default=0.10,
                    help="the max distance of softclip of one end to mask in head-to-tail mapping region. [Default: %(default)s]")
    P_search.add_argument("-ss", "--maxmaskallmissmap", type=float, default=0.35,
                    help="the max miss alignment distance in all sequence lenght. [Default: %(default)s]")
    P_search.add_argument("-dt", "--maxbpdistance", type=int, default=300,
                    help="the max distance of breakpoint of head-to-tail site. [Default: %(default)s]")
    P_search.add_argument("-mo", "--maxoverlap", type=int, default=1000,
                    help="the max overlap distance of head-to-tail region. [Default: %(default)s]")
    P_search.add_argument("-bo", "--maxneiboverlap", type=int, default=500,
                    help="the max overlap distance of neighbour region. [Default: %(default)s]")
    P_Search = subparsers.add_parser('Search',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_cnv],
                    help='search breakpoint region from bed file.')

    P_annot = subparsers.add_parser('Annot', conflict_handler='resolve', add_help=False)
    P_annot.add_argument("-at", "--annotefile",
                    help="the file used for region gene annotation.")
    P_annot.add_argument("-bi", "--biotype", nargs='+',
                    default=['miRNA','lncRNA', 'protein_coding'],
                    help="the gene biotype used for annotation of regions. [Default: %(default)s]")
    P_annot.add_argument("-kc", "--annotcol", type=str, default='gene_name',
                    help="the gene column used for annotation of regions. [Default: %(default)s]")
    P_annot.add_argument("-kt", "--annottype", type=str, default='bp', choices=['bp','all', 'part'],
                    help="the annotation regions. [Default: %(default)s]")
    P_annot.add_argument("-kl", "--annotbplen", type=int, default=5000000,
                    help="the length of two ends in BP to annotate when the annottype set as partial. [Default: %(default)s]")
    P_annot.add_argument("-ka", "--minoverlap", type=int, default=30,
                    help="the min overlap lenght between gene and link regioin. [Default: %(default)s]")
    P_annot.add_argument("-gb", "--gtfbed", type=str,
                    default='/share/home/share/Repository/GenomeDB/Reference/Homo_Sapiens/ENSEMBL/Homo_sapiens.GRCh38.100.gtf.gene.bed',
                    help="the gene bed file used for annotation of regions. [Default: %(default)s]")
    P_annot.add_argument("-sp", "--simplerepeat", type=str,
                    default='/share/home/share/Repository/GenomeDB/TandemRepeat/hg38_simpleRepeat.ensemb.bed',
                    help="the simplerepeat path. [Default: %(default)s]")
    P_annot.add_argument("-ko", "--minovertrf", type=int, default=30,
                    help="the min overlap between bed file and simplerepeat file. [Default: %(default)s]")
    P_annot.add_argument("-td", "--trfdistance", nargs=2, default=300,
                    help="the trf distance between bed file and simplerepeat file. [Default: %(default)s]")
    P_Annot = subparsers.add_parser('Annot',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_annot, P_cnv],
                    help='annotate region gene from bed file.')

    P_merge = subparsers.add_parser('Merge', conflict_handler='resolve', add_help=False)
    P_merge.add_argument("-ot", "--overtrimerrors", type=int, default=500,
                    help="the error margion bases in overlap link. [Default: %(default)s]")
    P_merge.add_argument("-rt", "--maxreadstwoends", type=int, default=500,
                    help="the max distance of breakpoint of two reads to merge. [Default: %(default)s]")
    P_merge.add_argument("-rw", "--readsmergeways", action='store_true', default=True,
                    help="whether to use the max distance of breakpoint to merge two reads region. [Default: %(default)s]")
    P_Merge = subparsers.add_parser('Merge',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_annot, P_merge, P_cnv],
                    help='merge breakpoint region from bed file.')

    P_update = subparsers.add_parser('Update', conflict_handler='resolve', add_help=False)
    P_update.add_argument("-ur", "--updatepre", type=str, default='All.circle.region',
                    help="out prefix of regioin out put.")
    P_Update = subparsers.add_parser('Update',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_merge, P_annot, P_update, P_cnv],
                    help='merge all breakpoint region in all samples.')

    P_filter = subparsers.add_parser('Filter', conflict_handler='resolve', add_help=False)
    P_filter.add_argument("-up", "--upmerge", type=str,
                    help="the update merge file path")
    P_filter.add_argument("-ch", "--Chrom", action='store_true', default=True,
                    help="only keep the specified chromosome: 1-22,X,Y,MT. [Default: %(default)s]")
    P_filter.add_argument("-bm", "--breakpiontnum", type=int, default=2,
                    help="the links with the threshlod of max breakpoint number in all samples. [Default: %(default)s]")
    P_filter.add_argument("-cv", "--maxcoverage" , type=float, default=0.4,
                    help="the max coverage in all samples on one link. [Default: %(default)s]")
    P_filter.add_argument("-dp", "--maxdepth" , type=float, default=0.85,
                    help="the max depth in all samples on one link. [Default: %(default)s]")
    P_filter.add_argument("-sn", "--minsupportnum", type=int, default=3,
                    help="the min support reads number in all samples on one link. [Default: %(default)s]")
    P_filter.add_argument("-hn", "--minhubnum", type=int, default=5,
                    help="the min support reads number in all samples on one hub. [Default: %(default)s]")
    P_filter.add_argument("-hs", "--minhubsize", type=int, default=3,
                    help="the min link number in the regioin defined as a hub. [Default: %(default)s]")
    P_filter.add_argument("-hl", "--minhublen", type=int, default=1000,
                    help="the min support reads number in the regioin defined as a hub. [Default: %(default)s]")
    P_filter.add_argument("-dn", "--minsidnum", type=int, default=1,
                    help="the min support reads id number in all samples on one link. [Default: %(default)s]")
    P_filter.add_argument("-ll", "--maxlinklen", type=int, default=10000000,
                    help="the max lenght on one link. [Default: %(default)s]")
    P_Filter = subparsers.add_parser('Filter',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_merge, P_update, P_annot, P_filter, P_cnv],
                    help='filter links from bed file.')

    P_circos = subparsers.add_parser('Circos', conflict_handler='resolve', add_help=False)
    P_circos.add_argument("-ci", "--circisin", type=str,
                    help="the circos input file. [Deflaut: UpFilterTRF].")
    P_circos.add_argument("-co", "--circisout", type=str,
                    help="the circos output directory. [Deflaut: Update dir].")
    P_circos.add_argument("-cs", "--circissw", type=str,
                    default='/share/home/share/software/circos-0.69-9/bin/circos',
                    help="the circos software. [Default: %(default)s].")
    P_circos.add_argument("-kb", "--keepbio", nargs='+',
                    default=['protein_coding'],
                    help="the kept gene biotype. [Default: %(default)s]")
    P_circos.add_argument("-pl", "--perl", type=str,
                    default='/share/home/share/software/Perl-5.32.1/bin/perl',
                    help="the perl software. [Default: %(default)s].")
    P_circos.add_argument("-hd", "--cirhead", type=str, default='All.circle.plot',
                    help="the circos output prefix name. [Default: %(default)s].")
    P_circos.add_argument("-gf", "--genefilt", type=str,
                    #default='/share/home/share/Pipeline/14EcDNA/EcDNAFinder/Data/cancer.gene.cosmic.cgc.723.20210225.txt',
                    help="the gene keep in circos. [Default: %(default)s].")
    P_circos.add_argument("-cp", "--cirplot", action='store_true', default=True,
                    help="whether to plot the circos. [Default: %(default)s]")
    P_Circos = subparsers.add_parser('Circos',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_merge, P_annot, P_update, P_filter, P_circos, P_cnv],
                    help='Circos visual for the target links file.')

    P_check = subparsers.add_parser('Check', conflict_handler='resolve', add_help=False)
    P_check.add_argument("-of", "--overfremin", type=float, default=0.8,
                    help="the minimum overlap ration of breakpiont region. [Default: %(default)s]")
    P_check.add_argument("-tt", "--bptotalrate", type=float, default=0.85,
                    help="the minimum ratio of breakpoint length  in total lenght. [Default: %(default)s]")
    P_check.add_argument("-tn", "--bptnum", type=int, default=2,
                    help="the min breakpoint number in one reads. [Default: %(default)s]")
    P_check.add_argument("-ol", "--overlenmin", type=int, default=400,
                    help="the minimum overlap lenght of breakpiont region. [Default: %(default)s]")
    P_check.add_argument("-cb", "--checkbed", type=str,
                    default='/share/home/zhou_wei/Workspace/11Project/02Plasmid/01analysescript/uniqueovr/BEDUniq.region.txt',
                    help="the bed file of plasmid unique region. [Default: %(default)s]")
    P_check.add_argument("-mc", "--maxchecksofttwoends", type=float, default=0.2,
                    help="the max distance of softclip of one end to mask in head-to-tail mapping region. [Default: %(default)s]")
    P_Check = subparsers.add_parser('Check',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_merge, P_check, P_cnv],
                    help='check plasmid unique breakpoint region.')

    P_seq = subparsers.add_parser('Seq', conflict_handler='resolve', add_help=False)
    P_seq.add_argument("-ls", "--lengthbpseq", type=int, default=1000,
                    help="the reference genome sequence legnth of breakpiont region to extract. [Default: %(default)s]")
    P_seq.add_argument("-gr", "--genome", type=str,
                    default='/share/home/zhou_wei/Workspace/01Repository/GenomeDB/Reference/EcDNARef/HG38_ENSEMBL_Plasmid20.fa',
                    help="the bed file of plasmid unique region. [Default: %(default)s]")
    P_seq.add_argument("-lf", "--linkfile", type=str,
                    help="the links file, such as All.circle.region.UpMerge_sort.")
    P_Seq = subparsers.add_parser('Seq',conflict_handler='resolve',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_merge, P_update, P_check, P_seq, P_cnv],
                    help='get sequence information.')

    P_Autopipe = subparsers.add_parser('Auto', conflict_handler='resolve', prefix_chars='-+',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    parents=[P_Common, P_fetch, P_search, P_merge, P_update, P_annot, P_filter, P_circos, P_check, P_cnv],
                    help='the auto-processing for all.')
    P_Autopipe.add_argument("+P", "++pipeline", nargs='+',
                    default=['CNV', 'Fetch', 'Search', 'Merge', 'Annot', 'Update', 'Filter', 'Circos'],
                    help="the auto-processing: [Default: %(default)s].")
    P_Autopipe.add_argument('+M','++MODEL' , nargs='+', type=str, default=['Standard'],
                    help='''Chose more the one models from Standard, Fselect,Fitting and Predict used for DIY pipline.''')
    args  = parser.parse_args()

    return args
