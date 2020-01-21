import numpy as np
import scipy.sparse as sp
import torch
import random
from sklearn.feature_extraction.text import TfidfTransformer


def clean_dblp(path='./data/dblp/',new_path='./data/dblp2/'):

    label_file = "author_label"
    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
                       dtype=np.int32)
    
    labels_raw = np.genfromtxt("{}{}.txt".format(path, label_file),
                               dtype=np.int32)
    
    A = {}
    for i,a in enumerate(labels_raw[:,0]):
        A[a]=i+1
    print(len(A))
    PA_new = np.asarray([[PA[i,0],A[PA[i,1]]] for i in range(PA.shape[0]) if PA[i,1] in A])
    PC_new = PC
    PT_new = PT

    labels_new = np.asarray([[A[labels_raw[i,0]],labels_raw[i,1]] for i in range(labels_raw.shape[0]) if labels_raw[i,0] in A])

    np.savetxt("{}{}.txt".format(new_path, PA_file),PA_new,fmt='%i')
    np.savetxt("{}{}.txt".format(new_path, PC_file),PC_new,fmt='%i')
    np.savetxt("{}{}.txt".format(new_path, PT_file),PT_new,fmt='%i')
    np.savetxt("{}{}.txt".format(new_path, label_file),labels_new,fmt='%i')

def gen_homograph():
    path = "data/dblp2/"
    out_file = "homograph"

    label_file = "author_label"
    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"
    APA_file = "APA"
    APAPA_file = "APAPA"
    APCPA_file = "APCPA"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                   dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                   dtype=np.int32)
    PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
                   dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1
    PT[:, 0] -= 1
    PT[:, 1] -= 1

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1]) + 1
    term_max = max(PT[:, 1]) + 1

    PA[:, 0] += author_max
    PC[:, 0] += author_max
    PC[:, 1] += author_max+paper_max

    edges = np.concatenate((PA,PC),axis=0)

    np.savetxt("{}{}.txt".format(path, out_file),edges,fmt='%u')

def read_embed(path="../../../data/dblp2/",
               emb_file="APC",emb_len=16):
    with open("{}{}_{}.emb".format(path, emb_file,emb_len)) as f:
        n_nodes, n_feature = map(int, f.readline().strip().split())
    print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))
    
    embedding = np.loadtxt("{}{}_{}.emb".format(path, emb_file,emb_len),
                           dtype=np.float32, skiprows=1)
    emb_index = {}
    for i in range(n_nodes):
        emb_index[embedding[i, 0]] = i

    features = np.asarray([embedding[emb_index[i], 1:] if i in emb_index else embedding[0, 1:] for i in range(18405)])

    #assert features.shape[1] == n_feature
    #assert features.shape[0] == n_nodes

    return features, n_nodes, n_feature

def dump_edge_emb(path='../../../data/dblp2/',emb_len=16):
    # dump APA
    APA_file = "APA"
    APAPA_file = "APAPA"
    APCPA_file = "APCPA"

    APA_e,n_nodes,n_emb =read_embed(path=path,emb_file='APC',emb_len=emb_len)
    APCPA_e,n_nodes,n_emb =read_embed(path=path,emb_file='APC',emb_len=emb_len)

    PA_file = "PA"
    PC_file = "PC"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1

    PAi={}
    APi={}
    PCi={}
    CPi={}

    for i in range(PA.shape[0]):
        p=PA[i,0]
        a=PA[i,1]

        if p not in PAi:
            PAi[p]=set()
        if a not in APi:
            APi[a]=set()

        PAi[p].add(a)
        APi[a].add(p)

    for i in range(PC.shape[0]):
        p=PC[i,0]
        c=PC[i,1]

        if p not in PCi:
            PCi[p]=set()
        if c not in CPi:
            CPi[c]=set()

        PCi[p].add(c)
        CPi[c].add(p)

    APAi={}
    APCi={}
    CPAi={}

    for v in APi:
        for p in APi[v]:
            if p not in PAi:
                continue
            for a in PAi[p]:
                if a not in APAi:
                    APAi[a] ={}
                if v not in APAi:
                    APAi[v] ={}

                if v not in APAi[a]:
                    APAi[a][v]=set()
                if a not in APAi[v]:
                    APAi[v][a]=set()

                APAi[a][v].add(p)
                APAi[v][a].add(p)
    
    for v in APi:
        for p in APi[v]:
            if p not in PCi:
                continue
            for c in PCi[p]:
                if v not in APCi:
                    APCi[v] ={}
                if c not in CPAi:
                    CPAi[c] ={}

                if c not in APCi[v]:
                    APCi[v][c]=set()
                if v not in CPAi[c]:
                    CPAi[c][v]=set()

                CPAi[c][v].add(p)
                APCi[v][c].add(p)



    ## APAPA; vpa1pa2
    #APAPA_emb = []
    #for v in APAi:
    #    result = {}
    #    count = {}
    #    for a1 in APAi[v]:
    #        np1 = len(APAi[v][a1])
    #        edge1 = [node_emb[p] for p in APAi[v][a1]]
    #        edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

    #        for a2 in APAi[a1].keys():
    #            np2 = len(APAi[a1][a2])
    #            edge2 = [node_emb[p] for p in APAi[a1][a2]]
    #            edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
    #            if a2 not in result:
    #                result[a2] = node_emb[a2] * (np2 * np1)
    #            else:
    #                result[a2] += node_emb[a2] * (np2 * np1)
    #            result[a2] += edge1 * np2
    #            result[a2] += edge2 * np1
    #            if a2 not in count:
    #                count[a2]=0
    #            count[a2] += np1*np2

    #    for a2 in result:
    #        if v <= a2:
    #            APAPA_emb.append(np.concatenate(([v, a2], result[a2]/count[a2], [count[a2]])))
    #APAPA_emb = np.asarray(APAPA_emb)
    #m = np.max(APAPA_emb[:, -1])
    #APAPA_emb[:, -1] /= m
    #print("compute edge embeddings {} complete".format('APAPA'))    

    APA_ps=sp.load_npz("{}{}".format(path, 'APA_ps.npz')).todense()
    APAPA_ps=sp.load_npz("{}{}".format(path, 'APAPA_ps.npz')).todense()
    APCPA_ps=sp.load_npz("{}{}".format(path, 'APCPA_ps.npz')).todense()

    # APA
    APA = APAi

    APA_emb = []
    for a1 in APA.keys():
        for a2 in APA[a1]:
            tmp = [APA_e[p] for p in APA[a1][a2]]
            tmp = np.sum(tmp, axis=0)/len(APA[a1][a2])
            tmp += APA_e[a1]+APA_e[a2]
            tmp /= 3
            if a1 <= a2:
                APA_emb.append(np.concatenate(([a1, a2], tmp,[APA_ps[a1,a2]], [len(APA[a1][a2])])))
    APA_emb = np.asarray(APA_emb)
    print("compute edge embeddings {} complete".format(APA_file))

    # APAPA
    APAPA_emb = []
    ind1 = APAi
    ind2 = APAi

    for v in ind1:
        result = {}
        count = {}
        for a1 in ind1[v].keys():
            np1 = len(ind1[v][a1])
            edge1 = [APA_e[p] for p in ind1[v][a1]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            for a2 in ind2[a1].keys():
                np2 = len(ind2[a1][a2])
                edge2 = [APA_e[p] for p in ind2[a1][a2]]
                edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
                if a2 not in result:
                    result[a2] = APA_e[a1] * (np2 * np1)
                else:
                    result[a2] += APA_e[a1] * (np2 * np1)
                result[a2] += edge1 * np2
                result[a2] += edge2 * np1
                if a2 not in count:
                    count[a2]=0
                count[a2] += np1*np2

        for a in result:
            if v <= a:
                APAPA_emb.append(np.concatenate(([v, a], (result[a]/count[a]+APA_e[a]+APA_e[v])/5
                                                 ,[APAPA_ps[v,a]],[count[a]])))
            # f.write('{} {} '.format(v, a))
            # f.write(" ".join(map(str, result[a].numpy())))
            # f.write('\n')
    APAPA_emb = np.asarray(APAPA_emb)
    m = np.max(APAPA_emb[:, -1])
    APAPA_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format(APAPA_file))

    #APCPA
    ind1 = APCi
    ind2 = CPAi
    APCPA_emb = []
    for v in ind1:
        result = {}
        count = {}
        if len(ind1[v]) == 0:
            continue
        for a1 in ind1[v].keys():
            np1 = len(ind1[v][a1])
            edge1 = [APCPA_e[p] for p in ind1[v][a1]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            for a2 in ind2[a1].keys():
                np2 = len(ind2[a1][a2])
                edge2 = [APCPA_e[p] for p in ind2[a1][a2]]
                edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
                if a2 not in result:
                    result[a2] = APCPA_e[a1] * (np2 * np1)
                else:
                    result[a2] += APCPA_e[a1] * (np2 * np1)
                if a2 not in count:
                    count[a2]=0
                result[a2] += edge1 * np2
                result[a2] += edge2 * np1
                count[a2] += np1*np2

        
        for a in result:
            if v <= a:
                if APCPA_ps[v,a]==0: print(v,a)
                APCPA_emb.append(np.concatenate(([v, a], (result[a]/count[a]+APCPA_e[a]+APCPA_e[v])/5,
                                                 [APCPA_ps[v,a]],
                                                 [count[a]])))
            # f.write('{} {} '.format(v,a))
            # f.write(" ".join(map(str, result[a].numpy())))
            # f.write('\n')
    APCPA_emb = np.asarray(APCPA_emb)
    m = np.max(APCPA_emb[:, -1])
    APCPA_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format(APCPA_file))
    emb_len=APA_emb.shape[1]-2
    np.savez("{}edge{}.npz".format(path, emb_len),
             APA=APA_emb, APAPA=APAPA_emb, APCPA=APCPA_emb)
    print('dump npz file {}edge{}.npz complete'.format(path, emb_len))
    pass

def pathsim(A):
    value = []
    x,y = A.nonzero()
    for i,j in zip(x,y):
        value.append(2 * A[i, j] / (A[i, i] + A[j, j]))
    return sp.coo_matrix((value,(x,y)))

def gen_homoadj():
    path = "data/dblp2/"

    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                   dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                   dtype=np.int32)
    PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
                   dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1
    PT[:, 0] -= 1
    PT[:, 1] -= 1

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1]) + 1
    term_max = max(PT[:, 1]) + 1

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                       shape=(paper_max, author_max),
                       dtype=np.float32)
    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, conf_max),
                       dtype=np.float32)
    #PT = sp.coo_matrix((np.ones(PT.shape[0]), (PT[:, 0], PT[:, 1])),
    #                   shape=(paper_max, term_max),
    #                   dtype=np.int32)

    APA = PA.transpose()*PA
    APAPA = APA*APA
    APCPA = PA.transpose()*PC * PC.transpose() * PA

    APA = pathsim(APA)
    APAPA = pathsim(APAPA)
    APCPA = pathsim(APCPA)

    sp.save_npz("{}{}".format(path, 'APA_ps.npz'), APA)
    sp.save_npz("{}{}".format(path, 'APAPA_ps.npz'), APAPA)
    sp.save_npz("{}{}".format(path, 'APCPA_ps.npz'), APCPA)

    #APA = np.hstack([APA.nonzero()[0].reshape(-1,1), APA.nonzero()[1].reshape(-1,1)])
    #APAPA = np.hstack([APAPA.nonzero()[0].reshape(-1,1), APAPA.nonzero()[1].reshape(-1,1)])
    #APCPA = np.hstack([APCPA.nonzero()[0].reshape(-1,1), APCPA.nonzero()[1].reshape(-1,1)])

    #np.savetxt("{}{}.txt".format(path, 'APA'),APA,fmt='%u')
    #np.savetxt("{}{}.txt".format(path, 'APAPA'),APA,fmt='%u')
    #np.savetxt("{}{}.txt".format(path, 'APCPA'),APA,fmt='%u')


def gen_walk(path='data/dblp2/'):
    APA_file = "APA"
    APAPA_file = "APAPA"
    APCPA_file = "APCPA"

    PA_file = "PA"
    PC_file = "PC"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1]) + 1

    PA[:, 0] += author_max
    PC[:, 0] += author_max
    PC[:, 1] += author_max+paper_max

    PAi={}
    APi={}
    PCi={}
    CPi={}

    for i in range(PA.shape[0]):
        p=PA[i,0]
        a=PA[i,1]

        if p not in PAi:
            PAi[p]=set()
        if a not in APi:
            APi[a]=set()

        PAi[p].add(a)
        APi[a].add(p)

    for i in range(PC.shape[0]):
        p=PC[i,0]
        c=PC[i,1]

        if p not in PCi:
            PCi[p]=set()
        if c not in CPi:
            CPi[c]=set()

        PCi[p].add(c)
        CPi[c].add(p)

    APAi={}
    APCi={}
    CPAi={}

    for v in APi:
        for p in APi[v]:
            if p not in PAi:
                continue
            for a in PAi[p]:
                if a not in APAi:
                    APAi[a] ={}
                if v not in APAi:
                    APAi[v] ={}

                if v not in APAi[a]:
                    APAi[a][v]=set()
                if a not in APAi[v]:
                    APAi[v][a]=set()

                APAi[a][v].add(p)
                APAi[v][a].add(p)
    
    for v in APi:
        for p in APi[v]:
            if p not in PCi:
                continue
            for c in PCi[p]:
                if v not in APCi:
                    APCi[v] ={}
                if c not in CPAi:
                    CPAi[c] ={}

                if c not in APCi[v]:
                    APCi[v][c]=set()
                if v not in CPAi[c]:
                    CPAi[c][v]=set()

                CPAi[c][v].add(p)
                APCi[v][c].add(p)

    #(1) number of walks per node w: 1000; TOO many
    #(2) walk length l: 100;
    #(3) vector dimension d: 128 (LINE: 128 for each order);
    #(4) neighborhood size k: 7; --default is 5
    #(5) size of negative samples: 5
    #mapping of notation: a:author v:paper i:conference
    l = 100
    w = 1000

    import random
    #gen random walk for meta-path APCPA
    with open("{}{}.walk".format(path,APCPA_file),mode='w') as f:
        for _ in range(w):
            for a in APi:
                #print(a)
                result="a{}".format(a)
                for _ in range(int(l/4)):
                    p = random.sample(APi[a],1)[0]
                    c = random.sample(PCi[p],1)[0]
                    result+=" v{} i{}".format(p,c)
                    p = random.sample(CPi[c],1)[0]
                    while p not in PAi:
                        p = random.sample(CPi[c],1)[0]
                    a = random.sample(PAi[p],1)[0]
                    result+=" v{} a{}".format(p,a)
                f.write(result+"\n")

    #gen random walk for meta-path APA
    with open("{}{}.walk".format(path,APA_file),mode='w') as f:
        for _ in range(w):
            for a in APi:
                result="a{}".format(a)
                for _ in range(int(l/2)):
                    p = random.sample(APi[a],1)[0]
                    a = random.sample(PAi[p],1)[0]
                    result+=" v{} a{}".format(p,a)
                f.write(result+"\n")
    ##gen random walk for meta-path APAPA
    #with open("{}{}.walk".format(path,APAPA_file),mode='w') as f:
    #    for _ in range(w):
    #        for a in APi:
    #            result="a{}".format(a)
    #            for _ in range(int(l/2)):
    #                p = random.sample(APi[a],1)[0]
    #                a = random.sample(PAi[p],1)[0]
    #                result+=" v{} a{}".format(p,a)
    #            f.write(result+"\n")
    
    pass

#clean_dblp()
#gen_homograph()
dump_edge_emb(emb_len=16)
#gen_homoadj()
#gen_walk()
