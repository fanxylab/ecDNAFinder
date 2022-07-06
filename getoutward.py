from collections import defaultdict
import pysam

filename='/share/home/zhou_wei/Workspace/11Project/03ecDNA/Nanopore/U2OS-gDNA/BamFile/BAM_Align/U2OS-gDNA_S8.redup.bam'


def read_pair_generator(bam, **kargs):
    """
    Generate read pairs in a BAM file or within a region string.
    Reads are added to read_dict until a pair is found.
    """
    read_dict = defaultdict(lambda: [None, None])
    for read in bam.fetch(**kargs):
        if ((not read.is_paired) or 
               read.is_secondary or
               #read.is_supplementary or
               read.is_unmapped or 
               read.reference_name != read.next_reference_name
        ):
            continue
        qname = read.query_name
        if qname not in read_dict:
            if read.is_read1:
                read_dict[qname][0] = read
            else:
                read_dict[qname][1] = read
        else:
            if read.is_read1:
                yield read, read_dict[qname][1]
            else:
                yield read_dict[qname][0], read
            del read_dict[qname]

inbam = pysam.AlignmentFile(filename, 'rb')
outward = pysam.AlignmentFile("./outward.bam", "wb", template=inbam)
for read1, read2 in read_pair_generator(inbam,  region=None):
    pos_fst = read2.reference_start - read1.reference_start
    is_fwd  = -1 if read1.is_reverse else 1
    is_mfwd = -1 if read2.is_reverse else 1
    if ( is_fwd*is_mfwd>0 ):
        '---<-R1-------<-R2-----'
        '----R1->-------R2->----'
        'otherward'
    elif pos_fst >0:
        '----R1---------R2------'
        if  is_fwd>0:
            '----R1->-------<-R2-----'
            'inward'
        else:
            '---<-R1---------R2->----'
            'outward'
            outward.write(read1)
            outward.write(read2)
    elif pos_fst <0:
        '----R2---------R1------'
        if is_fwd>0:
            '---<-R2---------R1->-----'
            'outward'
            outward.write(read1)
            outward.write(read2)
        else:
            '----R2->-------<-R1------'
            'inward'
inbam.close()
outward.close()
