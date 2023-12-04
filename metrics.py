import torchmetrics.functional as F
import torchmetrics.functional.classification as C

def get_metrics(y_hat, y, metric=None, thresh=0.5):
    if metric == "iou":
        score = C.binary_jaccard_index(y_hat, y, threshold=thresh)

    elif metric == "dice":
        score = F.dice(y_hat, y, threshold=thresh)

    elif metric == "recall":
        score = C.binary_recall(y_hat, y, threshold=thresh)
        
    return score