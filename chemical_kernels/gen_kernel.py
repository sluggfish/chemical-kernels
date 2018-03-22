
from .string_kernel import gws, mismatch
from .fingerprints import daylight_fps_tanimoto, morgan_fps_tanimoto
from .kernel_ridge_reg import evaluate_model

import numpy as np
import re


class Kernel():
    def __init__(self, kernel = None, y_true = None):
        self.check_size(kernel)
        self.kernel = kernel
        self.y_true = y_true
    
    @classmethod    
    def get_kernel(cls, path):
        kernel = []
        with open (path) as input:
            for line in input:
                # delete blank lines or lines of all 0's
                if re.match(r'^\s*$', line) or float(line.split(' ')[0]) - 0 < 0.00001:
                    continue
                kernel.append(list(map(float,line.strip().split(' '))))
        kernel = np.array(kernel)
        return (kernel)

    @classmethod
    def subseq_helper(cls, smiles, **kwargs):
        lamda = kwargs['lamda'] if 'lamda' in kwargs else None            
        p = kwargs['p'] if 'p' in kwargs else None
        tune = kwargs['p'] if 'tune' in kwargs else None

        if not tune:
            if lamda and p:
                K= gws(smiles, p, lamda)
            elif lamda:
                K = gws(smiles, 4, lamda)
            elif p:
                K = gws(smiles, p, 0.8)
            else:
                K = gws(smiles)
        else:   # tune both parameters
            K = gws(smiles, None, None, True)
            
        return (K)

    @classmethod
    def mismatch_helper(cls, smiles, **kwargs):
        k = kwargs['k'] if 'k' in kwargs else None
        m = kwargs['m'] if 'm' in kwargs else None

        if k and m:
            K = mismatch(smiles, k, m)
        elif k:
            K = mismatch(smiles, k, 1)
        elif m:
            K = mismatch(smiels, 4, m)
        else:
            K = mismatch(smiles)

        return (K)
    
    @classmethod        
    def fps_helper(cls, smiles, **kwargs):
        maxPath = kwargs['maxPath'] if 'maxPath' in kwargs else None

        if maxPath:
            K = daylight_fps_tanimoto(smiles, maxPath)
        else:
            K = daylight_fps_tanimoto(smiles)
            
        return (K)
    
    def check_size(self, K):
        if K.shape[0] != K.shape[1]:
            raise Exception("The Gram matrix doesn't have correct size.")

    @classmethod
    def from_file(cls, path, y_true):
        K = cls.get_kernel(path)
        return cls(K, y_true)
    
    @classmethod
    def from_smi(cls, smiles, y_true, kern_type, **kwargs):


        # string kernel
        if kern_type == 'subsequence':
            K = cls.subseq_helper(smiles, **kwargs)    
            return (cls(K, y_true))            
            
        # mismatch kernel
        if kern_type == 'mismatch':
            K = cls.mismatch_helper(smiles, **kwargs)
            return (cls(K, y_true))
        
        # daylight fingerprint 
        if kern_type == 'linear_fps':
            K = cls.fps_helper(smiles, **kwargs)  
            return (cls(K, y_true))
        
        if kern_type == 'morgan_fps':
            K = morgan_fps_tanimoto(smiles)
            return (cls(K, y_true))

        
    def accuracy(self):
        test_r2, r2 = evaluate_model(self.kernel, self.y_true)
        return (test_r2, r2)
    

