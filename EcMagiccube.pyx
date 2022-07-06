import numpy as np
import copy
cimport numpy as np

def neighbCigar(np.ndarray _G,  int errors=100):
    '''
    #chrom, start, end, forword, fflag, cigarreg
    '''
    cdef np.ndarray FLAG = _G[:,4].copy()
    cdef int _R = (<object> _G).shape[0]
    cdef np.ndarray _l, _k
    cdef int _n, _m

    if _R <2:
        return FLAG
    for _n in range(_R - 1):
        for _m in range(_n+1, _R):
            _l = _G[_n]
            _k = _G[_m]
            if _l[3] != _k[3]: continue
            if   (_l[-1][0] <= _k[-1][0] + errors) and (errors + _l[-1][1] >= _k[-1][1]):
                FLAG[_m] ='OVER'
            elif (_k[-1][0] <= _l[-1][0] + errors) and (errors + _k[-1][1] >= _l[-1][1]):
                FLAG[_n] ='OVER'
    return FLAG

def neighbBP(np.ndarray _G, int minneighbplen=500, int mingap=100, int maxneiboverlap=500):
    '''
    #chrom, start, end, forword, fflag, cigarreg
    '''
    cdef np.ndarray FLAG = _G[:,4].copy()
    cdef int _R = (<object> _G).shape[0]
    cdef np.ndarray BreakF, BreakL
    if _R <2:
        return FLAG
    for _n in range(_R - 1):
        BreakF = _G[_n]
        BreakL = _G[_n+1]

        if np.abs(BreakF[5][1]-BreakL[5][0]) > mingap  : continue
        if BreakF[5][1]-BreakF[5][0] < minneighbplen   : continue
        if BreakL[5][1]-BreakL[5][0] < minneighbplen   : continue
        if BreakF[0] != BreakL[0] : continue
        if BreakF[3] != BreakL[3] : continue

        if   (BreakF[1] > BreakL[1]) &\
                (BreakF[2] > BreakL[2]) &\
                (BreakF[1] >= BreakL[2] - maxneiboverlap) &\
                (BreakF[3] == '+'):

            FLAG[[_n, _n+1]] += ';EcBP;CiR' if (BreakF[1] <= BreakL[2]) else ';EcBP'

        elif (BreakF[1] < BreakL[1]) & \
                (BreakF[2] < BreakL[2]) & \
                (BreakL[1] >= BreakF[2] - maxneiboverlap) & \
                (BreakF[3] == '-'):
        
            FLAG[[_n, _n+1]] += ';EcBP;CiR' if (BreakL[1] <= BreakF[2]) else ';EcBP'
    return FLAG

def neighbBP2(np.ndarray _G, int minneighbplen=500, int maxcigarspan=800, int mingap=100, int maxneiboverlap=500):
    '''
    #chrom, start, end, forword, fflag, cigarreg
    '''
    cdef np.ndarray FLAG = _G[:,4].copy()
    cdef int _R = (<object> _G).shape[0]
    cdef np.ndarray BreakF, BreakL
    cdef int _n, _r

    if _R <2:
        return FLAG
    for _n in range(_R - 1):
        for _r in range(_n +1, _R):
            BreakF = _G[_n]
            BreakL = _G[_r]
            if BreakL[5][0] - BreakF[5][1]> maxcigarspan   : break
            if (_r -_n ==1) and (np.abs(BreakF[5][1]-BreakL[5][0]) > mingap) : continue
            if BreakF[5][1]-BreakF[5][0] < minneighbplen   : continue
            if BreakL[5][1]-BreakL[5][0] < minneighbplen   : continue
            if BreakF[0] != BreakL[0] : continue
            if BreakF[3] != BreakL[3] : continue

            if   (BreakF[1] > BreakL[1]) &\
                    (BreakF[2] > BreakL[2]) &\
                    (BreakF[1] >= BreakL[2] - maxneiboverlap) &\
                    (BreakF[3] == '+'):
                FLAG[[_n, _r]] += ';EcBP;CiR' if (BreakF[1] <= BreakL[2]) else ';EcBP'
                FLAG[_n+1: _r] = 'INTER'
                break

            elif (BreakF[1] < BreakL[1]) & \
                    (BreakF[2] < BreakL[2]) & \
                    (BreakL[1] >= BreakF[2] - maxneiboverlap) & \
                    (BreakF[3] == '-'):
                FLAG[[_n, _r]] += ';EcBP;CiR' if (BreakL[1] <= BreakF[2]) else ';EcBP'
                FLAG[_n+1: _r] = 'INTER'
                break
    return FLAG

def trimOverlink(np.ndarray _G,  int errors=500):
    '''
    '#chrom', 'start', 'end', 'forword', 'fflag'
    '''
    cdef np.ndarray FLAG = _G[:,-1].copy()
    cdef int _R = (<object> _G).shape[0]
    cdef list _N = []
    cdef np.ndarray _l, _k
    cdef int _n, _m

    if _R <2:
        return FLAG
    elif len(np.unique(_G[:,0])) > 1:
        FLAG += ';MultiChr'
        return FLAG
    else:
        for _n in range(_R-1):
            for _m in range(_n+1, _R):
                _l = _G[_n]
                _k = _G[_m]
                if _l[0] != _k[0]: continue
                if   (_l[1] <= _k[1] + errors) and (errors + _l[2] >= _k[2]):
                    if 'HTBREAKP' in FLAG[_m]:
                        _N.append(_n)
                    else:
                        _N.append(_m)
                elif (_k[1] <= _l[1] + errors) and (errors + _k[2] >= _l[2]):
                    if 'HTBREAKP' in FLAG[_n]:
                        _N.append(_m)
                    else:
                        _N.append(_n)

        FLAG[_N] +=';Trim'
        return FLAG

def CoverDepth(list Q, list R):
    cdef int L = R[1] - R[0] + 1
    cdef np.ndarray[np.int64_t, ndim=1] N = np.zeros(L, dtype=np.int64)
    cdef list i, j
    cdef float C, D

    for i in Q:
        for j in i :
            N[j[0]-R[0]:j[1]-R[0]+1] += 1

    C = N[N>0].shape[0]/len(N)
    D = sum(N)/len(N)
    return [C, D]

def InterV2(list intvs):
    """
    :param intvs: List[List[int]]
    :return: List[List[int]]
    """
    intvs = sorted(intvs)
    if  intvs[0][1] < intvs[1][0]:
        return intvs
    else:
        return [[ intvs[0][0], max(intvs[0][1], intvs[1][1])]]

def InterVs(list intvs):
    """
    :param intvs: List[List[int]]
    :return: List[List[int]]
    """
    cdef list Inters, merged, intv
    Inters = sorted(intvs, key=lambda x:x[0])
    merged = [ copy.deepcopy(Inters[0]) ]
    for intv in Inters[1:]:
        if  merged[-1][-1] < intv[0]:
            merged.append(intv)
        else:
            merged[-1][-1] = max(merged[-1][-1], intv[-1])
    return merged

def InterSm(list intvs):
    return sum(map(lambda x:x[1]-x[0], intvs))

def MaxBetween(np.ndarray inmap, int maxd):
    '''
    #row columns ['#chrom', 'start', 'end', 'forword', 'start_n', 'end_n']
    # input numpy must be sorted: inmap= inmap[np.lexsort((inmap[:,2], inmap[:,1]))]
    '''
    cdef np.ndarray l, B
    for l in inmap:
        B = (inmap[:,-2] >= l[1] - maxd) & (inmap[:,-2] <= l[1] + maxd) & \
            (inmap[:,-1] >= l[2] - maxd) & (inmap[:,-1] <= l[2] + maxd)
        if inmap[B].size!=0:
            inmap[B, -2] = inmap[B, -2].min()
            inmap[B, -1] = inmap[B, -1].max()
    return inmap

def OrderLinks(np.ndarray _G):
    ''''
    #row columns ['#chrom', 'start_n', 'end_n', 'length_n', 'forword', 'raw_order', 'query_name', 'SID', 'fflag']
    #add columns ['forword_n', 'Order', 'Link', 'LINKS']
    # input numpy must be sorted by: raw_order
    '''
    cdef np.ndarray _O
    cdef int _S = np.lexsort( (-_G[:,2],  _G[:,1], _G[:,0], -_G[:,3]) )[0]
    cdef int _R = (<object> _G).shape[0]

    if _G[_S, 4] == '+':
        _O = np.r_[_G[_S:], _G[:_S]]
        _O = np.c_[_O, _O[:,4]]   #forword_n
    else:
        _O = np.r_[_G[_S::-1], _G[:_S:-1]] 
        _O = np.c_[_O, np.vectorize({'+':'-','-':'+'}.get)(_O[:,4])] #forword_n
    del _G

    cdef list _T = [ '{0}:{1}-{2}'.format(*x[:3]) if x[-1] =='+' 
                        else '{0}:{2}-{1}'.format(*x[:3])
                    for x in _O[:,[0,1,2,-1]] ]  #apply_along_axis have a bug for str

    return np.column_stack(( _O,
                np.arange(1, _R+1), #Order
                np.array(_T),     #Link
                np.repeat(';'.join(_T), _R) )) #LINKS

def Transreg( list _c, long int _l=1000000):
    if (_c[2] - _c[1] >2*_l):
        return [_c[0], [[_c[1], _c[1]+_l], [_c[2]-_l, _c[2]]]]
    else:
        return [_c[0], [[_c[1], _c[2]]]]

def LinkAnnot( list _BPs, np.ndarray _genbed, int minover=30):
    cdef str _chr  = _BPs[0]
    cdef np.ndarray _tbed = _genbed[ (_genbed[:,0] == _chr) ]
    cdef int _S = (<object> _tbed).shape[0]
    cdef np.ndarray _tboo = np.zeros((_S),dtype=bool)
    cdef long int _start, _end
    for _start, _end in _BPs[1]:
        set1 = ((_tbed[:,1]>= _start) & (_tbed[:,2]<= _end))
        set2 = ((_tbed[:,1]< _start)  & (_tbed[:,2]>= _start+minover))
        set3 = ((_tbed[:,1]<= _end-minover) & (_tbed[:,2]> _end))
        _tboo += (set1 | set2 | set3)
    _tbed[_tboo,-2:]
    return [';'.join(_tbed[_tboo,-2]), ';'.join(_tbed[_tboo,-1])]

def BTAnnot( list _BP, np.ndarray _gbed, int mino=30, int trfd=300, int errlen=5):
    cdef np.ndarray _tbed = _gbed[(_gbed[:,0]==_BP[0])]
    cdef list _tboo = []
    cdef int _s, _e, i
    cdef bint set1, set2, set3
    for i in [1,2]:
        (_s, _e) =  (_BP[i], _BP[i]+trfd)  if (i==1) else (_BP[i]-trfd, _BP[i])
        set1 = ((_tbed[:,1]>= _s) & (_tbed[:,2]<=_e)).any()
        set2 = ((_tbed[:,1]<  _s) & (_tbed[:,2]>=_s + mino)).any()
        set3 = ((_tbed[:,1]<= _e - mino) & (_tbed[:,2] > _e)).any()
        _tboo.append( (set1 or set2 or set3) )
        #if (set1 or set2 or set3):
        #    _tboo.append( 'Htrf' if i==1 else 'Ttrf')
    #return ';'.join(_tboo)
    return _tboo

def EcRPM(np.ndarray _G, long int perdep=1000000):
    ''''
    #columns ['support_ID_num', 'mapped_reads']
    #mapped_reads also can be instead by sum of support_ID_num
    '''
    return perdep*_G[:,0]/_G[:,1]

def EcRPKM(np.ndarray _G, long int perdep=1000000, long int perlen=100000):
    ''''
    #columns ['support_ID_num', 'mapped_reads', 'LINKSLen']
    #mapped_reads also can be instead by sum of support_ID_num
    '''
    return perlen*perdep*_G[:,0]/(_G[:,1]*_G[:,2])

def EcTPM(np.ndarray _G, long int perdep=1000000, long int perlen=100000):
    ''''
    #columns ['support_ID_num', 'LINKSLen']
    '''
    cdef np.ndarray A = perlen*_G[:,0]/_G[:,1]
    return (perdep*A)/sum(A)

def GroupBY(np.ndarray _K, str _M='EcTPM', long int col=0, ):
    cdef int _R = (<object> _K).shape[0]
    cdef np.ndarray[np.double_t, ndim=1]  _T = np.zeros(_R)
    cdef np.ndarray idx
    cdef str i
    for i in np.unique( _K[:,col] ):
        idx = _K[:,col] == i
        _T[idx] = eval( _M, globals())(_K[idx, 1:])
    return _T
