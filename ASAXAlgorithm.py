import numpy as np
import Util
import SAXRep
from numba import jit
import math


@jit(nopython=True)
def MiSAX_ENTmN(timeSeries,nb_segments,alphabet_size,seg_lim):
    nb_timeSeries=timeSeries.shape[0]
    ts_len=timeSeries.shape[1]
    minn = np.min(timeSeries)
    #print("min=", minn)
    maxx = np.max(timeSeries)
    #print("max=", maxx)
    cuts = Util.cuts_ENT(minn, maxx, alphabet_size)
    #print(cuts)
    index=ts_len//2
    indexes=np.array([0,index,ts_len])
    #print(indexes)

    isaxRepresentationSaved = list()

    for i in range(len(timeSeries)):
        ts_PAA = Util.PAA_varSegSize(timeSeries[i], indexes)
        #print(ts_PAA)
        isaxRepresentationSaved.append(SAXRep.saxRep(ts_PAA, cuts))
    #print("here")
    #print(isaxRepresentationSaved)
    k = 2
    while(k!=nb_segments):
        #print("****")
        #print("indexes",indexes)
        #print("****")
        indexSplitPosition=0
        h=0
        #print(k)
        for i in range(k):
            #print("i=",i)
            newIndex = indexes[i] + ((indexes[i+ 1] - indexes[i]) // 2)
            if((newIndex-indexes[i])>seg_lim):
                indexes_temp = np.array((indexes[i],newIndex,indexes[i+1]))
                #print("index temp",indexes_temp)
                table={}

                for j in range(len(timeSeries)):
                    #paa
                    ts_PAA = Util.PAA_varSegSize(timeSeries[j], indexes_temp)
                    #print(ts_PAA)
                    #isaxrep
                    isaxWord=SAXRep.saxRep(ts_PAA, cuts)
                    #print(isaxRepresentationSaved[j])
                    isaxWord=np.concatenate((np.append(isaxRepresentationSaved[j][:i], isaxWord), isaxRepresentationSaved[j][i + 1:]))
                    isaxWordStr=SAXRep.toStrUsingChr(isaxWord)

                    #print("******* word :",isaxWordStr)
                    # calcul occurence
                    #isaxWordStr=hash(isaxWordStr)
                    if isaxWordStr in table:
                        table[isaxWordStr]+=1
                    else:
                        table[isaxWordStr] = 1
                occs=np.array(list(table.values()))
                #print(occs)
                #calcul entropie
                ent=Util.entropy(occs,nb_timeSeries)
                #print(occs,"entropie ",ent)
                if(ent>h):
                    h=ent
                    indexSplitPosition=i

        #print("index split position :",indexSplitPosition)
        newIndex=indexes[indexSplitPosition]+((indexes[indexSplitPosition+1]-indexes[indexSplitPosition])//2)
        #print("new index :",newIndex)
        indexes_temp = np.array((indexes[indexSplitPosition], newIndex, indexes[indexSplitPosition + 1]))
        indexes=np.concatenate((np.append(indexes[:indexSplitPosition+1],newIndex),indexes[indexSplitPosition+1:]))

        for j in range(len(timeSeries)):
            ts_PAA = Util.PAA_varSegSize(timeSeries[j], indexes_temp)
            isaxWord = SAXRep.saxRep(ts_PAA, cuts)
            isaxRepresentationSaved[j] = np.concatenate((np.append(isaxRepresentationSaved[j][:indexSplitPosition], isaxWord), isaxRepresentationSaved[j][indexSplitPosition + 1:]))
        k=k+1
        #print("fin")

    return indexes


def iSAXOcc(timeSeries,nb_segments,alphabet_size):
    table={}
    cuts = SAXRep.getBreakPoints(alphabet_size)
    for ts in timeSeries:
        # paa
        ts_PAA = Util.PAA_fixedSegSize(ts,nb_segments)
        #print(ts_PAA)
        # isaxrep
        isaxWord = SAXRep.saxRep(ts_PAA, cuts)
        isaxWordStr = SAXRep.toStr(isaxWord)
        print(isaxWordStr)
        # calcul occurence
        if isaxWordStr in table:
            table[isaxWordStr] += 1
        else:
            table[isaxWordStr] = 1
    occs = np.array(list(table.values()))
    print(occs)
    return occs





