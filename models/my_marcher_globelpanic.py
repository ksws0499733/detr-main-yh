# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import nested_tensor_from_tensor_list

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_iou: float = 1,cost_mask: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_iou = cost_iou
        self.cost_mask = cost_mask

        assert cost_class != 0 or cost_bbox != 0 or cost_iou != 0 or cost_mask != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries,h,w = outputs["pred_masks"].shape


        # We flatten to compute the cost matrices in a batch
        # -------------------------------flatten(startAxis, endAxis)
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        out_mask = outputs["pred_masks"].flatten(0, 1)  # [batch_size * num_queries, W/4,H/4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_mask = targer_cat(targets).to(out_mask)

        tgt_mask = torch.nn.functional.interpolate(tgt_mask[:, None], size=(h,w),
                        mode="bilinear", align_corners=False)[:, 0]
        # tgt_mask = tgt_mask[:, 0]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]  # batch_size * num_queries X batch_size * num_targets

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        cost_mask = mask_iou_cost_dice(out_mask, tgt_mask)
        

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_iou * cost_giou + self.cost_mask*cost_mask
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] #for each batch, ?????????????????????????????????
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

def mask_iou_cost_dice(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    inputs = inputs.sigmoid()  # B,H,W
    numerator = torch.einsum("ic,jc->ij", inputs, targets) *2
    # numerator = 2 * (inputs * targets).sum(1)
    inputs_sum = inputs.sum(-1)
    targets_sum = targets.sum(-1)

    denominator = inputs_sum[:,None] + targets_sum[None,:]
    cost = 1 - (numerator + 1) / (denominator + 1)
    return cost


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def targer_cat(targets):
    masks_list =[v["masks"] for v in targets] 
    c, h, w = _max_by_axis([list(img.shape) for img in masks_list])  # N*W*H

    dtype = masks_list[0].dtype
    device = masks_list[0].device

    c_size = [img.shape[0] for img in masks_list]
    tensor_list = [torch.zeros((ci,h,w), dtype=dtype, device=device) for ci in c_size]

    for img, pad_img in zip(masks_list, tensor_list):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    
    return torch.cat(tensor_list)
    


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, 
                        cost_bbox=args.set_cost_bbox, 
                        cost_iou=args.set_cost_giou,
                        cost_mask=args.set_cost_mask)
