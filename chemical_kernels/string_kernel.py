import numpy as np
import math


# Gap-weighted subsequences kernels
# Input: strings s and t of lengths n and m, length p, weight λ
def gws_kern(mol1, mol2, p, lamda):
    n = len(mol1)
    m = len(mol2)
    DPS = np.zeros((n, m))
    kern = [0]*p
    
    for i in range(n):
        for j in range(m):
            if mol1[i] == mol2[j]:
                DPS[i,j] = lamda**2
        
    kern[0] = np.sum(DPS)
    DP = np.zeros((n-1, m-1))
    for l in range(1,p):
        DP[0,0] = DPS[0,0]
        # boundary values
        for i in range(1,n-1):
            DP[i,0] = DPS[i,0] + lamda * DP[i-1,0]
        for j in range(1,m-1):
            DP[0,j] = DPS[0,j] + lamda * DP[0,j-1]
            
        # update DP
        for i in range(1,n-1):
            for j in range(1,m-1):
                DP[i,j] = DPS[i,j] + lamda * DP[i-1,j] + lamda * DP[i,j-1] + (lamda**2) * DP[i-1,j-1]
            
        # update DPS and kernel value
        for i in range(1,n):
            for j in range(1,m):
                if mol1[i] == mol2[j]:
                    DPS[i,j] = (lamda**2) * DP[i-1,j-1]
                    kern[l] = kern[l] + DPS[i,j]
                    
    return (kern)


def gws_matrix(smiles, p, lamda):
    kern = [[] for _ in range(p)]
    for i in range(len(smiles)):
        temp = gws_kern(smiles[i],smiles[i],p,lamda)
        for l in range(p):
            # compute square root here
            kern[l].append(math.sqrt(temp[l]))
    
    sim = [[] for _ in range(p)]
    for i in range(len(smiles)):
        for l in range(p):
            sim[l].append([])
        for j in range(len(smiles)):
            temp = gws_kern(smiles[i],smiles[j],p,lamda)
            for l in range(p):
                # normalize
                sim[l][i].append(temp[l]/(kern[l][i]*kern[l][j]))
                
    str_kern = np.array(sim)
    return (str_kern)


def gws(smiles, p = 4, lamda = 0.8, tune = False):
    if not tune:
        gram = gws_matrix(smiles, p, lamda)[p-1]
        return (gram)
   

def mismatch_count(smiles, k = 5, m = 1):
    # construct the alphabet for smiles representation of mols
    alphabet = set()
    for smile in smiles:
        for c in smile:
            alphabet.add(c)

    # compute the dict for counting characters (l=1)
    N = len(smiles)
    t1 = {i:{} for i in alphabet}
    for x in alphabet:
        for i in range(N):
            for j in range(len(smiles[i])):
                t1[x][(i,j)] = 0 if smiles[i][j] == x else 1
    
    prev = t1
    
    for i in range(1, k - 1):
        cur = {}
        for key in prev:
            for c in alphabet:                
                temp = tuple(list(key) + [c])  # potential substrings
                for key2 in prev[key]:
                    i, j = key2
                    if prev[key][key2] <= m and j <= len(smiles[i]) - 2:                
                        if temp not in cur:
                            cur[temp] = {}
                        count = prev[key][key2] if smiles[i][j+1] == c else prev[key][key2] + 1  # count mismatches
                        if count <= m:
                            cur[temp][(i,j+1)] = count
        prev = cur
                            
    count_tree = {}
    for key in prev:
        for c in alphabet:
            temp = tuple(list(key) + [c])
            for key2 in prev[key]:
                i, j = key2
                if prev[key][key2] <= m and j <= len(smiles[i]) - 2:                
                    if temp not in count_tree:
                        count_tree[temp] = {}
                    if i not in count_tree[temp]:
                        count_tree[temp][i] = {}
                    count = prev[key][key2] if smiles[i][j+1] == c else prev[key][key2] + 1
                    if count <= m:
                        count_tree[temp][i][j+1] = count
    return (count_tree)                        


def mismatch_matrix(count_tree, N):
    # compute gram matrix                    
    K = np.zeros((N,N))
    for key1 in count_tree:
        l = count_tree[key1].keys()
        for i in l:
            for j in l:
                K[i, j] += len(count_tree[key1][i].keys()) * len(count_tree[key1][j].keys())
                
    # normalize
    gram = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            gram[i,j] = K[i,j] / math.sqrt(K[i,i] * K[j,j])
            
    return (gram)


# mismatch kernel
# k for length of compared substrings, m for # of mismatches allowed
def mismatch(smiles, k = 4, m = 1):
    N = len(smiles)
    count_tree = mismatch_count(smiles, k, m)
    gram = mismatch_matrix(count_tree, N)
    return (gram)

