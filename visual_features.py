from torch.jit.annotations import Tuple, List, Dict, Optional
from torchvision.ops import boxes as box_ops
import torchvision
from tqdm.notebook import tqdm 
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pickle
import os 


def postprocess_detections_custom(self, class_logits, box_regression, proposals, image_shapes):
    # type: (Tensor, Tensor, List[Tensor], List[Tuple[int, int]])
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    all_keeps = []

    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[:self.detections_per_img]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        all_keeps.append(keep)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels, all_keeps


def roi_heads_customed(self, features, proposals, image_shapes):
    """
    """

    box_features = self.box_roi_pool(features, proposals, image_shapes)
    box_features = self.box_head(box_features)

    class_logits, box_regression = self.box_predictor(box_features)
    result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])

    boxes, scores, labels, keep = postprocess_detections_custom(self, class_logits, box_regression, proposals, image_shapes)
    
    return box_features[keep]


def customed_forward(model, images):
    """
    """

    original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, None)
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([('0', features)])
    proposals, proposal_losses = model.rpn(images, features, targets)
    roi_features = roi_heads_customed(model.roi_heads, features, proposals, images.image_sizes)

    return roi_features 


def build_features(data, model):
    features = {}

    for elt in tqdm(data):
        path = elt['img']
        img = plt.imread(f'data/{path}')
        with torch.set_grad_enabled(False):
            image_features = customed_forward(model_frcnn, torch.tensor(img[:, :, :3]).permute(2, 0, 1).unsqueeze(0).to(device))
            features[elt['id']] = image_features.cpu()

    pickle.dump(features, open(f'image_features/images_features_dict.pkl', 'wb'))


if __name__ == '__main__':

    model_frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model_frcnn.eval()
    device = torch.device("cuda")
    model_frcnn = model_frcnn.to(device)

    if not os.path.exists('images_features'):
        os.makedirs('images_features')

    train_set = []
    with jsonlines.open('data/train.jsonl', 'r') as f:
      for line in f: 
        train_set.append(line)

    dev_set = []
    with jsonlines.open('data/dev.jsonl', 'r') as f:
      for line in f: 
        dev_set.append(line)

    test_set = []
    with jsonlines.open('data/test.jsonl', 'r') as f:
      for line in f: 
        test_set.append(line)

    data = train_set + dev_set + test_set 

    build_features(data, model_frcnn)
