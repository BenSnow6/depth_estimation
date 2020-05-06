import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math


def get_depth(depth_img,u,v):
    return depth_img[u][v]
    
def isValid(depth_1, depth_2):
    return (depth_1>0 and depth_2>0)

def calc_errors(pred_depth, grndt_depth):
    if (pred_depth.shape[0] == grndt_depth.shape[0] and pred_depth.shape[1] == grndt_depth.shape[1]):
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
        print("Depth maps do not have the same dimensions!")
    return [difference_err, sqr_diff_err, inv_err, inv_sqr_err, log_err, log_sqr_err, scale_inv_err, abs_rel_err, sqr_rel_err]