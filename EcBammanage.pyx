from pysam.libcalignedsegment cimport AlignedSegment
from pysam.libcalignmentfile cimport AlignmentFile

cdef class BamFilter():
    cdef dict arg
    cdef public list Head, samlist
    cdef public long int Counts
    def __cinit__(self, dict arg):
        self.arg  = arg

    cpdef cigarmerge(self, list _ct, str match='Q'):
        cdef long int indel = self.arg['maskindel'] #100000
        cdef long int skip  = self.arg['maskskip']  #10000000
        cdef long int hard  = self.arg['maskhard']  #100000
        cdef long int pad   = self.arg['maskpad']   #10000000
        cdef bint softfq    = self.arg['getsoftfq'] #False
        cdef list ct  = [ (0, i[1]) if (i[0] in [7,8]) else i for i in _ct ]
        cdef list ctN = ct[:1]

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

    def bamfilter(self, str inbam, str inid):
        cdef long int minsoflt  = self.arg['minsoftdrop']
        cdef long int lensoftfq = self.arg['lensoftfq']
        cdef int mapq = self.arg['minmapQ']
        cdef AlignmentFile samfile = AlignmentFile(inbam, "rb")

        def filter_read(AlignedSegment read):            
            return not ( read.is_secondary
                        or read.is_unmapped
                        or read.is_duplicate
                        or read.is_qcfail
                        or read.mapping_quality < mapq )
                        #or read.query_length - read.query_alignment_length < minsoflt
        cdef list Info=[]
        cdef list raw_cigartuples
        cdef str  is_reverse

        self.Head = ['#chrom', 'start', 'end', 'SID', 'length', 'forword', 'query_name', 'query_length',
                        'query_alignment_length', 'mapping_quality', 'flag', 'cigarreg',  'cigarstring']
        self.samlist = []
        self.Counts  = 0
        for read in samfile.fetch():
            if filter_read(read):
                if read.flag in [0, 1, 16]:
                    self.Counts += 1
                if read.query_length - read.query_alignment_length >= minsoflt:
                    raw_cigartuples = read.cigartuples         
                    is_reverse = '-' if read.is_reverse else '+'
                    Info = [read.reference_name, read.reference_start, read.reference_end, inid, 
                            read.reference_end - read.reference_start + 1, is_reverse, read.query_name, 
                            read.query_length, read.query_alignment_length, read.mapping_quality, read.flag ]
                    if is_reverse =='-':
                        read.cigartuples = raw_cigartuples[::-1]
                    Info.append(tuple([read.qstart, read.qend]))

                    read.cigartuples = self.cigarmerge(raw_cigartuples, match='Q')
                    Info.append( read.cigarstring )

                    self.samlist.append(Info)
        samfile.close()
        return self

    def mapcounts(self, str inbam, int mapq=0):
        cdef AlignmentFile samfile = AlignmentFile(inbam, "rb")
        def filter_read(AlignedSegment read):            
            return not ( read.is_secondary
                        or read.is_unmapped
                        or read.is_duplicate
                        or read.is_qcfail
                        or read.mapping_quality < mapq )
        #cdef set mapset = set()
        cdef dict mapset = {}
        for read in samfile.fetch():
            if filter_read(read):
                #mapset.add(read.query_name)
                mapset[read.query_name] = 0
        samfile.close()
        return len(mapset)