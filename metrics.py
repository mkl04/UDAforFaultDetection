import numpy as np
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt


def compute_iou(preds, target):
    """
    Computes iou for binary classification problems
    
    Parameters:
        preds (torch.tensor): predicted labels
        target (torch.tensor): true labels

    Returns:
        iou (float): IOU score
    """
    
    target = target.view(-1)
    pred = preds.view(-1)

    cls = 1
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu() - intersection

    if union == 0:
        iou = np.nan
    else:
        iou = float(intersection) / float(max(union, 1))
    
    return iou


def binary_classification_metrics(probs, target, threshold=0.5):
    """
    Computes precision, recall, and F1 score for binary classification problems
    
    Parameters:
        probs (torch.tensor): predicted labels in probs
        target (torch.tensor): true labels
        threshold (float): threshold

    Returns:
        precision (float): precision score
        recall (float): recall score
        f1_score (float): f1 score
    """
    preds = (probs > threshold).long() # convert predictions to binary format
    tp = (preds * target).sum().float() # true positive count
    fp = (preds * (1 - target)).sum().float() # false positive count
    fn = ((1 - preds) * target).sum().float() # false negative count

    precision = tp / (tp + fp + 1e-8) # add a small constant to avoid divide-by-zero errors
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision.item(), recall.item(), f1_score.item()


# compute distance from all pixels set to 0
def computeDistanceFromFault(img):
    """
    Compute distance image from all original fault pixels set to 1.

    Parameters:
        img: input image (with value=1 for fault and value=0 for background)

    Returns:
        distance image
    """
    #invert pixel values for distance computation
    out = distance_transform_edt(1-img)
    return out


def computeMetric(im1, im2, sigma=5):
    """
    Common mean function used by precision and recall

    Parameters:
        im1: input image
        im2: input image

    Returns:
        mean value
    """
    ones = np.ones(im1.shape)
    zeros = np.zeros(im1.shape)
    # search pixel where limits exists
    n = np.where(im1 >0, ones, zeros)
    # compute f(d_im2) where limits exists
    d = np.where(im1 >0, np.exp(-im2*im2/(2*sigma*sigma)), zeros)
    return np.sum(d)/np.sum(n)


def getPrecisionRecallTable(gt, pred, sigma):
    """
    Compute precision and recall

    Parameters:
        gt: ground truth image
        pred: prediction image
        sigma: list of tolerance distance values

    Returns:
        list of precision and recall values
    """
    # compute distance from ground truth
    d_gt = computeDistanceFromFault(gt)
    # compute distance from prediction
    d_pred = computeDistanceFromFault(pred)
    lsigma = np.array([])
    precision = np.array([])
    recall = np.array([])
    for i in sigma:
        #add sigma to the list of tolerance distance 
        lsigma = np.append(lsigma, i)
        # compute precision and append the result to a list
        precision = np.append(precision, computeMetric(pred, d_gt, sigma=i))
        #compute recall and append the result to a list
        recall = np.append(recall, computeMetric(gt, d_pred, sigma=i))        
    
    # compute IOU from precision and recall
    iou = 1/(1/recall+1/precision-1)
    f1 = 2/(1/recall+1/precision)
    
    return lsigma, f1, iou


def getNewMetrics(true_imgs, pred_imgs, sigma, wandb=None):
    """
    Compute new f1 score and IOU

    Parameters
        true_imgs: list of true images
        pred_imgs: list of predicted images
        sigma: tolerance distance value
        wandb: logger
    """
    
    new_iou = []
    new_f1 = []
    for gt, pred in tqdm(zip(true_imgs, pred_imgs)):
        
        # compute distance from ground truth
        d_gt = computeDistanceFromFault(gt)
        # compute distance from prediction
        d_pred = computeDistanceFromFault(pred)

        # compute precision and append the result to a list
        precision = computeMetric(pred, d_gt, sigma=sigma)
        # compute recall and append the result to a list
        recall = computeMetric(gt, d_pred, sigma=sigma)       

        # compute IOU from precision and recall
        iou = 1/(1/recall+1/precision-1)
        f1 = 2/(1/recall+1/precision)

        new_iou.append(iou)
        new_f1.append(f1)
        
    new_iou = np.nanmean(new_iou)
    new_f1 = np.nanmean(new_f1)

    if wandb:
        wandb.log({f"valid/iou_tgt/sigma-{sigma}": new_iou})
        wandb.log({f"valid/f1_tgt/sigma-{sigma}": new_f1})
    
    print("With Sigma={:.0f}:  IoU: {:.4f}   F1-score: {:.4f}".format(sigma, new_iou, new_f1 ))