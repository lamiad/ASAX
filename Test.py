import Util
import ASAXAlgorithm
import time

file="ECGFiveDays"
n=861
m=130


data=Util.readDataset(file,n,m)

timeSeries=Util.ds_normalization(data)

nb_segment=m//10

card=32

seg_lim=1 #specify min length


queriesPath="Example/queries "+file+".txt"


queries=Util.queryFileToTS(queriesPath)
nb_queries=len(queries)



nb_NN=10
app_KNN_SAX=list()
app_KNN_ENT=list()

GT=list()

for query in queries:

    GT.append(Util.GT_KNN_Search(timeSeries,query,nb_NN))

print("######################################## SAX")

means_precision=list()

for query in queries:
    query_PAA=Util.PAA_fixedSegSize(query,nb_segment)
    NN = dict()
    id=0
    for ts in timeSeries:
        ts_PAA = Util.PAA_fixedSegSize(ts,nb_segment)
        dist = Util.DR(query_PAA,ts_PAA,m)
        if len(NN)<nb_NN:
            NN[dist]=(id,ts)
        else:
            maxDist = max(NN)
            if dist < maxDist:
                del NN[maxDist]
                NN[dist]=(id,ts)

        id+=1

    app_KNN_SAX.append(Util.App_KNN_Search(NN.values(),query))





for K_NN in (2,4,6,8,10):
    s=0
    for i in range(nb_queries):
        p = Util.accuracyPC(GT[i][:K_NN],app_KNN_SAX[i][:K_NN],K_NN)
        s += p

    mean = s / nb_queries
    print("mean for K_NN= ", K_NN,": ",mean)
    means_precision.append(mean)





print("######################################## ASAX")


means_precision=list()

indexes=ASAXAlgorithm.MiSAX_ENTmN(timeSeries, nb_segment, card, seg_lim)
start_time=time.time()
indexes=ASAXAlgorithm.MiSAX_ENTmN(timeSeries, nb_segment, card, seg_lim)
exec_time=time.time()-start_time
print("exec time :",exec_time)

segs_len=Util.segments_len(indexes)

for query in queries:
    query_PAA=Util.PAA_varSegSize(query,indexes)
    NN = dict()
    id=0
    for ts in timeSeries:
        ts_PAA = Util.PAA_varSegSize(ts,indexes)
        dist = Util.DR_VAR(query_PAA,ts_PAA,segs_len)
        if len(NN)<nb_NN:
            NN[dist]=(id,ts)
        else:
            maxDist = max(NN)
            if dist < maxDist:
                del NN[maxDist]
                NN[dist]=(id,ts)

        id+=1

    app_KNN_ENT.append(Util.App_KNN_Search(NN.values(),query))




for K_NN in (2,4,6,8,10):
    s=0
    for i in range(nb_queries):

        p = Util.accuracyPC(GT[i][:K_NN],app_KNN_ENT[i][:K_NN],K_NN)
        s += p

    mean = s / nb_queries
    print("mean for K_NN= ", K_NN,": ",mean)
    means_precision.append(mean)



exit(0)
