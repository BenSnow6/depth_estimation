import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import sys
from tqdm.notebook import trange, tqdm
import torch
import torchvision
import pandas as pd
import os
from os import walk
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from os import listdir
from os.path import isfile, join
import torch.nn.functional as F
from torchsummary import summary
import math
import pickle


def get_depth(depth_img,u,v):
    return depth_img[u][v]
    
def isValid(depth_1, depth_2):
    return (depth_1>0 and depth_2>0)

class MyError(Exception):
    pass

def calc_errors(pred_depth, grndt_depth):
    difference_err = 0  # 1
    sqr_diff_err = 0    # 2
    inv_err = 0         # 3
    inv_sqr_err = 0     # 4
    log_err = 0         # 5
    log_sqr_err = 0     # 6
    log_non_abs_err = 0 # 7
    abs_rel_err = 0     # 8
    sqr_rel_err = 0     # 9
    valid_pixels = 0   # valid pixel count
    if (pred_depth.shape[0] == grndt_depth.shape[0] and pred_depth.shape[1] == grndt_depth.shape[1]):
        for u in range(0, pred_depth.shape[0]):
            for v in range(0, pred_depth.shape[1]):
                # grabbing the depths at point (u,v) from the depth maps
                depth_of_pred = get_depth(pred_depth,u,v)
                depth_of_gt = get_depth(grndt_depth,u,v)
                if isValid(depth_of_pred,depth_of_gt):          # check non negative depth

                    # calculate the absolute difference between depth maps for a given pixel
                    diff = abs(depth_of_gt-depth_of_pred)                               
                    # squared difference
                    sqr = diff*diff                                                     
                    # absolute inverse difference
                    inv = abs(1/depth_of_gt-1/depth_of_pred)                           
                    # squared inverse difference
                    inv_sqr = inv*inv
                    # absolute log difference
                    log_diff_abs = abs(math.log(depth_of_gt)-math.log(depth_of_pred))   
                    # absolute log square difference
                    log_sqr = log_diff_abs*log_diff_abs                                 
                    # log difference (non absolute)
                    log_diff = math.log(depth_of_gt) - math.log(depth_of_pred)




                    difference_err += diff                      # increment difference error                                      # 1
                    sqr_diff_err += sqr                         # increment square difference error                               # 2
                    inv_err += inv                              # increment inverse difference error                              # 3
                    inv_sqr_err += inv_sqr                      # increment inverse square difference error                       # 4
                    log_err += log_diff_abs                     # increment absolute log error                                    # 5         
                    log_sqr_err += log_sqr                      # increment absolute square log error                             # 6   
                    log_non_abs_err += log_diff                 # increment log difference error (for scale invariant log error)  # 7    
                    abs_rel_err += diff/depth_of_gt             # increment absolute relative error                               # 8       
                    sqr_rel_err += sqr/(depth_of_gt*depth_of_gt) # increment square relative error                       # 9

                    valid_pixels += 1                           # increment number of valid pixels




        # Normalisation
        # normalising to give mean average error (mae)
        difference_err = difference_err/valid_pixels
        # norm and sqrt for root mean square error (rmse)
        sqr_diff_err = sqr_diff_err/valid_pixels
        sqr_diff_err = math.sqrt(sqr_diff_err)
        # norm inverse abs error (iae)
        inv_err = inv_err/valid_pixels
        # norm and sqrt for inverse rmse (irmse)
        inv_sqr_err = inv_sqr_err/valid_pixels
        inv_sqr_err = math.sqrt(inv_sqr_err)
        # norm log mean absolute error
        log_err = log_err/valid_pixels
        # save normed square log error as it is required for scale invariant error
        norm_sqr_log = log_sqr_err/valid_pixels
        # sqrt to give log rmse
        log_sqr_err = math.sqrt(norm_sqr_log)
        # scale invariant error
        scale_inv_err = math.sqrt(norm_sqr_log - (log_non_abs_err*log_non_abs_err)/(valid_pixels*valid_pixels))
        # norm absolute relative error
        abs_rel_err = abs_rel_err/valid_pixels
        # norm relative square error
        sqr_rel_err = sqr_rel_err/valid_pixels

    else:
        raise MyError('Depth maps do not have the same dimensions!')
    return [difference_err, sqr_diff_err, inv_err, inv_sqr_err, log_err, log_sqr_err, scale_inv_err, abs_rel_err, sqr_rel_err]

def predict_and_gt(_dl, _val_size, _batch_size, _model):

    count_test_batches = 0
    chunk_size = _val_size//_batch_size
    for xb, yb in tqdm(_dl, leave=False):
        if count_test_batches == 0:
            # initialise list of predicted depths
            numpy_depth_prediction = [None]*(chunk_size+1)
            gt_depths = [None]*(chunk_size+1)
            for i in range(0, (chunk_size+1)):
                numpy_depth_prediction[i] = [None]*xb.shape[0]
                gt_depths[i] = [None]*xb.shape[0]
        # transform data into floats and then place on gpu
        xb, yb = xb.float(), yb.float()
        xb, yb = xb.cuda(), yb.cuda()

        # run the x's through the trained network to predict their depth maps
        prediction = _model(xb)
        
        # save each predicted depth map in the batch to a list as a numpy array
        for i in range(0, xb.shape[0]):
            # reshape the predictions to the correct size:(720,1280) from (1,720,1280)
            numpy_depth_prediction[count_test_batches][i] = np.reshape(prediction[i].cpu().detach().numpy(), (xb.shape[2],xb.shape[3]))
            # save ground truth depths to list
            gt_depths[count_test_batches][i] = yb[i].cpu().detach().numpy()
        count_test_batches += 1
    return numpy_depth_prediction, gt_depths

def mean_and_std_errors(predictions, grnd_trth, val_size, batch_size):

    error_dictionary = {}
    chunk_size = val_size//batch_size
    for i in range(chunk_size+1):
        error_dictionary[f"{i}"] = {}
    for i in range(chunk_size+1):
        for j in range(batch_size):
            error_dictionary[f"{i}"][f"{j}"] = {}

    #//////////////////////////////////////#

    # calculate the errors

    for i in range(chunk_size-1):
        for j in range(batch_size-1):
            error_dictionary[f"{i}"][f"{j}"] = calc_errors(predictions[i][j], grnd_trth[i][j])
        print(f"Calculating errors for batch {i} of {int(chunk_size)}")

    #//////////////////////////////////////#

    # initialisation of average errors
    difference_err_avg  = 0
    sqr_diff_err_avg    = 0
    inv_err_avg         = 0
    inv_sqr_err_avg     = 0
    log_err_avg         = 0
    log_sqr_err_avg     = 0
    log_non_abs_err_avg = 0
    abs_rel_err_avg     = 0
    sqr_rel_err_avg     = 0

    #//////////////////////////////////////#

    # increment the errors

    for i in range(0, chunk_size-1):
        for j in range(0, batch_size-1):
            difference_err_avg  += error_dictionary[f"{i}"][f"{j}"][0]
            sqr_diff_err_avg    += error_dictionary[f"{i}"][f"{j}"][1]
            inv_err_avg         += error_dictionary[f"{i}"][f"{j}"][2]
            inv_sqr_err_avg     += error_dictionary[f"{i}"][f"{j}"][3]
            log_err_avg         += error_dictionary[f"{i}"][f"{j}"][4]
            log_sqr_err_avg     += error_dictionary[f"{i}"][f"{j}"][5]
            log_non_abs_err_avg += error_dictionary[f"{i}"][f"{j}"][6]
            abs_rel_err_avg     += error_dictionary[f"{i}"][f"{j}"][7]
            sqr_rel_err_avg     += error_dictionary[f"{i}"][f"{j}"][8]
        print(f"Incrimenting for batch {i} of {chunk_size-1}")

    #//////////////////////////////////////#

    ## divide by number of images to get average error
    difference_err_avg  /= (val_size)
    sqr_diff_err_avg    /= (val_size)
    inv_err_avg         /= (val_size)
    inv_sqr_err_avg     /= (val_size)
    log_err_avg         /= (val_size)
    log_sqr_err_avg     /= (val_size)
    log_non_abs_err_avg /= (val_size)
    abs_rel_err_avg     /= (val_size)
    sqr_rel_err_avg     /= (val_size)

    #//////////////////////////////////////#

    # initialise difference counters
    difference_err_count    = 0
    sqr_diff_err_count      = 0
    inv_err_count           = 0
    inv_sqr_err_count       = 0
    log_err_count           = 0
    log_sqr_err_count       = 0
    log_non_abs_err_count   = 0
    abs_rel_err_count       = 0
    sqr_rel_err_count       = 0

    #//////////////////////////////////////#

    # sum squared differences
    for i in range(0, chunk_size-1):
        for j in range(0, batch_size-1):
            difference_err_count    += (error_dictionary[f"{i}"][f"{j}"][0] - difference_err_avg)**2
            sqr_diff_err_count      += (error_dictionary[f"{i}"][f"{j}"][1] - sqr_diff_err_avg)**2
            inv_err_count           += (error_dictionary[f"{i}"][f"{j}"][2] - inv_err_avg)**2
            inv_sqr_err_count       += (error_dictionary[f"{i}"][f"{j}"][3] - inv_sqr_err_avg)**2
            log_err_count           += (error_dictionary[f"{i}"][f"{j}"][4] - log_err_avg)**2
            log_sqr_err_count       += (error_dictionary[f"{i}"][f"{j}"][5] - log_sqr_err_avg)**2
            log_non_abs_err_count   += (error_dictionary[f"{i}"][f"{j}"][6] - log_non_abs_err_avg)**2
            abs_rel_err_count       += (error_dictionary[f"{i}"][f"{j}"][7] - abs_rel_err_avg)**2
            sqr_rel_err_count       += (error_dictionary[f"{i}"][f"{j}"][8] - sqr_rel_err_avg)**2
        print(f"Sum sqr for batch {i} of {chunk_size-1}")
        

    #//////////////////////////////////////#

    # divide by number of test images
    difference_err_count    /= val_size
    sqr_diff_err_count      /= val_size
    inv_err_count           /= val_size
    inv_sqr_err_count       /= val_size
    log_err_count           /= val_size
    log_sqr_err_count       /= val_size
    log_non_abs_err_count   /= val_size
    abs_rel_err_count       /= val_size
    sqr_rel_err_count       /= val_size

    #//////////////////////////////////////#

    # square root
    difference_err_sigma    = math.sqrt(difference_err_count)
    sqr_diff_err_sigma      = math.sqrt(sqr_diff_err_count)
    inv_err_sigma           = math.sqrt(inv_err_count)
    inv_sqr_err_sigma       = math.sqrt(inv_sqr_err_count)
    log_err_sigma           = math.sqrt(log_err_count)
    log_sqr_err_sigma       = math.sqrt(log_sqr_err_count)
    log_non_abs_err_sigma   = math.sqrt(log_non_abs_err_count)
    abs_rel_err_sigma       = math.sqrt(abs_rel_err_count)
    sqr_rel_err_sigma       = math.sqrt(sqr_rel_err_count)

    #//////////////////////////////////////#

    # collect and return errors and std deviations
    mean_errors = [difference_err_avg, sqr_diff_err_avg, inv_err_avg, inv_sqr_err_avg, log_err_avg, log_sqr_err_avg, log_non_abs_err_avg, abs_rel_err_avg, sqr_rel_err_avg]
    std_devs    = [difference_err_sigma, sqr_diff_err_sigma, inv_err_sigma, inv_sqr_err_sigma, log_err_sigma, log_sqr_err_sigma, log_non_abs_err_sigma, abs_rel_err_sigma, sqr_rel_err_sigma]
    return mean_errors, std_devs

def load_calculated_errors():

    path_saved = f'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/val_means.pckl' ## Change this path to the directory where the predictions are stored, don't change the file name from val_means.pckl
    f = open(path_saved, 'rb')
    val_means = pickle.load(f)
    f.close()
    path_saved = f'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/val_stds.pckl' ## Change this path to the directory where the predictions are stored, don't change the file name from val_stds.pckl
    f = open(path_saved, 'rb')
    val_stds = pickle.load(f)
    f.close()
    path_saved = f'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/test_means.pckl' ## Change this path to the directory where the predictions are stored, don't change the file name from test_means.pckl
    f = open(path_saved, 'rb')
    test_means = pickle.load(f)
    f.close()
    path_saved = f'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/test_stds.pckl'## Change this path to the directory where the predictions are stored, don't change the file name from test_stds.pckl
    f = open(path_saved, 'rb')
    test_stds = pickle.load(f)
    f.close()

    return val_means, val_stds, test_means, test_stds

def load_test_preds_and_gts():

    path_saved = 'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/test_preds.pckl' ## Change this path to the directory where the predictions are stored, don't change the file name from test_preds.pckl
    f = open(path_saved, 'rb')
    test_preds = pickle.load(f)
    f.close()
    path_saved = 'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/test_gts.pckl' ## Change this path to the directory where the predictions are stored, don't change the file name from test_gts.pckl
    f = open(path_saved, 'rb')
    test_gts = pickle.load(f)
    f.close()

    return test_preds, test_gts

def load_val_preds_and_gts():

    path_saved = 'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/val_preds.pckl' ## Change this path to the directory where the predictions are stored, don't change the file name from val_means.pckl
    f = open(path_saved, 'rb')
    val_preds = pickle.load(f)
    f.close()
    path_saved = 'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/val_gts.pckl' ## Change this path to the directory where the predictions are stored, don't change the file name from val_gts.pckl
    f = open(path_saved, 'rb')
    val_gts = pickle.load(f)
    f.close()
    
    return val_preds, val_gts

def load_calculated_kitti_errors():

    path_saved = 'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/kitti_means.pckl' ## Change filepath but not file name
    f = open(path_saved, 'rb')
    kitti_means = pickle.load(f)
    f.close()
    
    path_saved = 'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/kitti_stds.pckl' ## Change filepath but not file name
    f = open(path_saved, 'rb')
    kitti_stds = pickle.load(f)
    f.close()

    return kitti_means , kitti_stds

def load_kitti_preds_and_gts():
    
    path_saved = 'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/kitti_preds.pckl' ## Change filepath but not file name
    f = open(path_saved, 'rb')
    kitti_preds = pickle.load(f)
    f.close()
    
    path_saved = 'C:/Users/Ben/OneDrive - Bournemouth University/Computer Vision/Datasets/Saved_preds/kitti_gts.pckl' ## Change filepath but not file name
    f = open(path_saved, 'rb')
    kitti_gts = pickle.load(f)
    f.close()

    return kitti_preds , kitti_gts