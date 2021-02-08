from numba import jit
import numpy as np
import ASAXAlgorithm
import SAXRep
import csv
import matplotlib.pyplot as plt
import random
#data = np.array([12.0, 13.0, 6.0, 2.0],dtype=float)


def normalization(ts):
    ts_mean=np.mean(ts)
    #print(ts_mean)
    ts_std=np.std(ts)
    #print(ts_std)
    if ts_std==0:
        print("std = 0")
    ts_normalized=np.empty_like(ts)
    #print(ts_normalized)
    for i in range(len(ts)):
        ts_normalized[i]=(ts[i]-ts_mean)/ts_std
    return ts_normalized

def ds_normalization(timesSeries):
    timeSeries_normalized = np.empty((timesSeries.shape[0], timesSeries.shape[1]))
    i = 0
    for ts in timesSeries:
        timeSeries_normalized[i] = normalization(ts)
        i += 1
    return timeSeries_normalized
@jit(nopython=True)
def seg_mean(ts,start,end):
    sum=0
    #print(start," hta ",end)
    for i in range(start,end+1):
        sum+=ts[i]
    #print(sum/(end-start+1))
    return sum/(end-start+1)

def segs_mean(timeSeries):
    nb_segments=timeSeries.shape[1]
    mean=np.empty(nb_segments)
    for i in range(nb_segments):
        mean[i]=np.mean(timeSeries[:,i])
    return mean

@jit(nopython=True)
def stdv(timeSeries):
    nb_segments = timeSeries.shape[1]
    stdev = np.empty(nb_segments)
    for i in range(nb_segments):
        stdev[i]=np.std(timeSeries[:,i])
    return stdev

def eucDistance(q,c):
    return np.sqrt(np.sum((q-c)**2))

def DR(q_PAA,c_PAA,ts_len):
    return np.sqrt(ts_len/len(q_PAA))*np.sqrt(np.sum((q_PAA-c_PAA)**2))

def DR_VAR(q_PAA,c_PAA,segs_len):
    sum = 0
    for i in range(len(q_PAA)):
        sum += ((q_PAA[i]-c_PAA[i]) ** 2) * (segs_len[i])

    return np.sqrt(sum)

def MINDIST(q_s,c_s,ts_len,cuts):
    return np.sqrt(ts_len/len(q_s))*np.sqrt(sum_dist(q_s,c_s,cuts))

def sum_dist(q_s,c_s,cuts):
    sum=0
    for i in range(len(q_s)):
        sum+=(dist(q_s[i],c_s[i],cuts))**2
    return sum

def dist(r,c,cuts):
    r=int(r)
    c=int(c)
    if abs(r-c)<=1:
        return 0
    else:
        return cuts[max(r,c)-1]-cuts[min(r,c)]

def MINDIST_VAR(q_s,c_s,segs_len,cuts):
    return np.sqrt(sum_dist_var(q_s,c_s,segs_len,cuts))

def sum_dist_var(q_s,c_s,segs_len,cuts):
    sum=0
    for i in range(len(q_s)):
        sum+=((dist(q_s[i],c_s[i],cuts))**2)*(segs_len[i])
    return sum

def segments_len(indexes):
    w=len(indexes)-1
    segs_len=np.empty(w,dtype=int)
    for i in range(w):
        segs_len[i]=indexes[i+1]-indexes[i]
    return segs_len



def PAA_fixedSegSize(ts,nb_segments):

    ts_len=len(ts)
    segment_size=ts_len/nb_segments
    ts_PAA=np.empty(nb_segments)
    if nb_segments!=ts_len:

        offset = 0
        for i in range(nb_segments):
            ts_PAA[i]=seg_mean(ts,offset,offset+int(segment_size)-1)
            #print("paa i ",ts_PAA[i])
            offset+=int(segment_size)
    else:
        ts_PAA=np.copy(ts)

    return ts_PAA
@jit(nopython=True)
def PAA_varSegSize(ts,indexes):

    nb_segments=len(indexes)-1
    ts_PAA = np.empty(nb_segments)

    for i in range(nb_segments):
        ts_PAA[i]=seg_mean(ts,indexes[i],indexes[i+1]-1)

    return ts_PAA

@jit(nopython=True)
def entropy(occs,db_size):
    h=0
    p=occs/db_size
    #print(p)
    logp=np.log2(p)
    for i in range(len(occs)):
        h=np.sum(p*logp)
    return -h

def ds_entropyiSAX(timeSeries,nb_segments,alphabet_size):
    return entropy(ASAXAlgorithm.iSAXOcc(timeSeries, nb_segments, alphabet_size), timeSeries.shape[0])

def ds_entropyMiSAX(timeSeries,nb_segments,alphabet_size):
    occ=ASAXAlgorithm.MiSAXOcc(timeSeries, nb_segments, alphabet_size)
    return entropy(occ,timeSeries.shape[0])


def readDataset(file,n,m):
    # nb line 40 million
    # nb column 200
    file = open("Example/"+file+".txt","r")
    timeSeries=np.empty((n,m))
    for i in range(n):
        ts_StrValues=file.readline().split(",")
        #print(ts_StrValues)
        #print(i,"   ",len(ts_StrValues))

        for j in range(m):
            timeSeries[i,j]=float(ts_StrValues[j+1])
    file.close()
    return timeSeries

def lenNN(NN):
    nb=0
    for k in NN:
        nb+=len(NN[k])
    return nb

def toRemove(NNv,query):
    i=0
    max=eucDistance(query,NNv[0][1])
    for j in range(1,len(NNv)):
        dist=eucDistance(query,NNv[j][1])
        if dist>max:
            max=dist
            i=j
    return i


def StrToTS(line):
    tsStr = line.split(",")
    ts = np.empty(len(tsStr), dtype=float)
    for j in range(len(tsStr)):
        ts[j] = np.float(tsStr[j])
    return ts

def queryFileToTS(path):
    timeSeries=list()
    file=open(path,"r")
    line=file.readline()
    while line!='':
        #print(line)
        ts=StrToTS(line)
        timeSeries.append(ts)
        line=file.readline()
    return timeSeries


def accuracyPC(id_GT,id_App_KNN,K_NN):
    total=0
    for id in id_App_KNN:
        if id in id_GT:
            total+=1

    return total/K_NN


def App_KNN_Search(NN_values,query):
    newList=list(NN_values)
    newList.sort(key= lambda x:eucDistance(query,x[1]))

    id_App_KNN=list()
    for e in newList:
        id_App_KNN.append(e[0])
    return id_App_KNN


def GT_KNN_Search(timeSeries,query,nb_NN):
    NN = dict()
    id=0
    for ts in timeSeries:
        dist = eucDistance(query,ts)
        if dist in NN.keys():
            raise Exception("double")
        if len(NN)<nb_NN:
            NN[dist]=(id,ts)
        else:
            maxDist = max(NN)
            if dist < maxDist:
                    del NN[maxDist]
                    NN[dist]=(id,ts)
        id+=1

    newList = list(NN.values())

    newList.sort(key=lambda x: eucDistance(query, x[1]))

    id_GT_KNN = list()
    for e in newList:
        id_GT_KNN.append(e[0])
    return id_GT_KNN



def tsvToTxt(file,out):
    lines_seen = set()  # holds lines already seen
    outfile = open("Example/"+out+".txt", "w")
    tsv_file = open(file)

    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        line = str(row[0])
        for i in range(1, len(row)):
            line += "," + str(row[i])
        if line not in lines_seen:  # not a duplicate
            outfile.write(line + "\n")
            lines_seen.add(line)
    outfile.close()

def chooseQ(file,n,m,nb_queries,qfile):

    randomList = random.sample(range(0, n), nb_queries)
    randomList.sort()
    print(randomList)
    print(len(randomList))

    data = readDataset(file, n, m)
    timeSeries = ds_normalization(data)

    fQueriesPath = "Example/"+qfile+".txt"

    f = open(fQueriesPath, "w")

    for n in randomList:
        newEntry = str(timeSeries[n][0])
        for i in range(1, len(timeSeries[n])):
            newEntry += "," + str(timeSeries[n][i])
        f.write(newEntry + "\n")
    f.close()
@jit(nopython=True)
def cuts_ENT(min,max,card):
    offset=(max-min)/card
    cuts=np.empty(card-1)
    cuts[0]=min+offset
    for i in range(1,len(cuts)):
        cuts[i]=cuts[i-1]+offset
    return cuts

