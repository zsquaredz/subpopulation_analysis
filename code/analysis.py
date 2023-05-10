from json.tool import main
import os, sys
from matplotlib import pyplot as plt
# matplotlib inline
import numpy as np
import pickle
import pandas
import gzip
import argparse
import cca_core

def SVCCA(file1, file2, dim1_to_keep, dim2_to_keep, mask_file, use_mask=False):
    acts1 = np.load(file1) # data points x number of hidden dimension 
    acts2 = np.load(file2)
    
    acts1 = np.float32(acts1)
    acts2 = np.float32(acts2)
    acts1 = acts1.T
    acts2 = acts2.T
    # print(acts1.shape) # need to be (number of neurons, number of test data points)
    # print(acts2.shape)

    # Mean subtract activations
    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    
    print('starting to perform SVD')
    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False) # s1: min(row, col) V1 min(row, col) x data points size 
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    s1_sq = [i*i for i in s1]
    s2_sq = [i*i for i in s2]

    
    print("Fraction of variance explained by", dim1_to_keep ,"singular vectors for input1", np.sum(s1_sq[:dim1_to_keep])/np.sum(s1_sq))
    print("Fraction of variance explained by", dim2_to_keep ,"singular vectors for input2", np.sum(s2_sq[:dim2_to_keep])/np.sum(s2_sq))
    
    
    svacts1 = np.dot(s1[:dim1_to_keep]*np.eye(dim1_to_keep), V1[:dim1_to_keep]) # s1[:20]*np.eye(20) 20 x 20      V1[:20]   20 x number of data points  --> 20 x number of genreal tokens
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    # this will become dim1_to_keep x number of data points 
    svacts2 = np.dot(s2[:dim2_to_keep]*np.eye(dim2_to_keep), V2[:dim2_to_keep])
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)

    if use_mask:
        with open(mask_file) as f:
            word_mask_list = []
            for line in f.readlines():
                word_mask_list += [int(x) for x in line.strip().split()]
            word_mask = np.array(word_mask_list, dtype=bool)
            assert len(word_mask) == svacts1.shape[1] # sanity check
            assert len(word_mask) == svacts2.shape[1] # sanity check
            svacts1 = svacts1[:,word_mask]
            svacts2 = svacts2[:,word_mask]
    if use_mask:
        print(mask_file.split('.')[-1], 'mask has been applied, SVD done')
    else:
        print('SVD done')


    # print('starting to perform CCA')
    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-8, verbose=False)
    print("result", np.mean(svcca_results["cca_coef1"]))


def Corr(file1, file2):
    attentions1 = np.load(file1)
    attentions2 = np.load(file2)
    corr = np.corrcoef(attentions1, attentions2)
    print("Corr result", corr[0][1])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data")
    parser.add_argument("--category1", type=str, default='Books', help="category")
    parser.add_argument("--data_dir1", type=str, default='./data/', help="Directory of data")
    parser.add_argument("--category2", type=str, default='Books', help="category")
    parser.add_argument("--data_dir2", type=str, default='./data/', help="Directory of data")
    parser.add_argument("--do_svcca", action='store_true', help="Whether to do SVCCA")
    parser.add_argument("--do_corr", action='store_true', help="Whether to do correlation")
    parser.add_argument("--use_mask", action='store_true', help="Whether to use the provided mask to apply to the data")
    parser.add_argument("--mask_dir", type=str, default='./data/', help="Directory of mask")
    parser.add_argument("--svd_dim1", type=int, default=750, help="Dimensions to keep after SVD")
    parser.add_argument("--svd_dim2", type=int, default=750, help="Dimensions to keep after SVD")
    args = parser.parse_args()
    
    if args.do_svcca:
        SVCCA(args.data_dir1, args.data_dir2, args.svd_dim1, args.svd_dim2, args.mask_dir, args.use_mask)
    elif args.do_corr:
        Corr(args.data_dir1, args.data_dir2)
    

    