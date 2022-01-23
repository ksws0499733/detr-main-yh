# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import io
import unittest

import torch
from torch import nn, Tensor
from typing import List

from models.matcher import HungarianMatcher
from models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from models.backbone import Backbone, Joiner, BackboneBase
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
from hubconf import detr_resnet50, detr_resnet50_panoptic

# onnxruntime requires python 3.5 or above
try:
    import onnxruntime
except ImportError:
    onnxruntime = None


class Tester(unittest.TestCase):

    # def test_box_cxcywh_to_xyxy(self):
    #     t = torch.rand(10, 4)
    #     r = box_ops.box_xyxy_to_cxcywh(box_ops.box_cxcywh_to_xyxy(t))
    #     self.assertLess((t - r).abs().max(), 1e-5)

    # @staticmethod
    # def indices_torch2python(indices):
    #     return [(i.tolist(), j.tolist()) for i, j in indices]

    # def test_hungarian(self):
    #     n_queries, n_targets, n_classes = 100, 15, 91
    #     logits = torch.rand(1, n_queries, n_classes + 1)
    #     boxes = torch.rand(1, n_queries, 4)
    #     tgt_labels = torch.randint(high=n_classes, size=(n_targets,))
    #     tgt_boxes = torch.rand(n_targets, 4)
    #     matcher = HungarianMatcher()
    #     targets = [{'labels': tgt_labels, 'boxes': tgt_boxes}]
    #     indices_single = matcher({'pred_logits': logits, 'pred_boxes': boxes}, targets)
    #     indices_batched = matcher({'pred_logits': logits.repeat(2, 1, 1),
    #                                'pred_boxes': boxes.repeat(2, 1, 1)}, targets * 2)
    #     self.assertEqual(len(indices_single[0][0]), n_targets)
    #     self.assertEqual(len(indices_single[0][1]), n_targets)
    #     self.assertEqual(self.indices_torch2python(indices_single),
    #                      self.indices_torch2python([indices_batched[0]]))
    #     self.assertEqual(self.indices_torch2python(indices_single),
    #                      self.indices_torch2python([indices_batched[1]]))

    #     # test with empty targets
    #     tgt_labels_empty = torch.randint(high=n_classes, size=(0,))
    #     tgt_boxes_empty = torch.rand(0, 4)
    #     targets_empty = [{'labels': tgt_labels_empty, 'boxes': tgt_boxes_empty}]
    #     indices = matcher({'pred_logits': logits.repeat(2, 1, 1),
    #                        'pred_boxes': boxes.repeat(2, 1, 1)}, targets + targets_empty)
    #     self.assertEqual(len(indices[1][0]), 0)
    #     indices = matcher({'pred_logits': logits.repeat(2, 1, 1),
    #                        'pred_boxes': boxes.repeat(2, 1, 1)}, targets_empty * 2)
    #     self.assertEqual(len(indices[0][0]), 0)

    # def test_position_encoding_script(self):
    #     m1, m2 = PositionEmbeddingSine(), PositionEmbeddingLearned()
    #     mm1, mm2 = torch.jit.script(m1), torch.jit.script(m2)  # noqa

    # def test_backbone_script(self):
    #     backbone = Backbone('resnet50', True, False, False)
    #     torch.jit.script(backbone)  # noqa

    # def test_model_script_detection(self):
    #     model = detr_resnet50(pretrained=False).eval()
    #     scripted_model = torch.jit.script(model)
    #     x = nested_tensor_from_tensor_list([torch.rand(3, 128, 96), 
    #                                         # torch.rand(3, 128, 96),
    #                                         torch.rand(3, 128, 96)])
    #     print("==s1==")
    #     out = model(x)
    #     # print("==s2==")
    #     # out_script = scripted_model(x)
    #     print("==1==",out["pred_logits"].shape)
    #     print("==2==",out["pred_boxes"].shape)
    #     # self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
    #     # self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))

    def test_model_script_panoptic(self):
        model = detr_resnet50_panoptic(pretrained=False).eval()
        # scripted_model = torch.jit.script(model)
        x = nested_tensor_from_tensor_list([torch.rand(3, 128, 96), torch.rand(3, 128, 96)])
        out = model(x)
        print("==1==",out["pred_logits"].shape)
        print("==2==",out["pred_boxes"].shape)
        print("==3==",out["pred_masks"].shape)
        # out_script = scripted_model(x)
        # self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
        # self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))
        # self.assertTrue(out["pred_masks"].equal(out_script["pred_masks"]))

    # def test_model_detection_different_inputs(self):
    #     model = detr_resnet50(pretrained=False).eval()
    #     # support NestedTensor
    #     x = nested_tensor_from_tensor_list([torch.rand(3, 200, 200), torch.rand(3, 200, 250)])
    #     out = model(x)
    #     self.assertIn('pred_logits', out)
    #     # and 4d Tensor
    #     x = torch.rand(1, 3, 200, 200)
    #     out = model(x)
    #     self.assertIn('pred_logits', out)
    #     # and List[Tensor[C, H, W]]
    #     x = torch.rand(3, 200, 200)
    #     out = model([x])
    #     self.assertIn('pred_logits', out)

    # def test_warpped_model_script_detection(self):
    #     class WrappedDETR(nn.Module):
    #         def __init__(self, model):
    #             super().__init__()
    #             self.model = model

    #         def forward(self, inputs: List[Tensor]):
    #             sample = nested_tensor_from_tensor_list(inputs)
    #             return self.model(sample)

    #     model = detr_resnet50(pretrained=False)
    #     wrapped_model = WrappedDETR(model)
    #     wrapped_model.eval()
    #     scripted_model = torch.jit.script(wrapped_model)
    #     x = [torch.rand(3, 200, 200), torch.rand(3, 200, 250)]
    #     out = wrapped_model(x)
    #     out_script = scripted_model(x)
    #     self.assertTrue(out["pred_logits"].equal(out_script["pred_logits"]))
    #     self.assertTrue(out["pred_boxes"].equal(out_script["pred_boxes"]))


@unittest.skipIf(onnxruntime is None, 'ONNX Runtime unavailable')
class ONNXExporterTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(self, model, inputs_list, tolerate_small_mismatch=False, do_constant_folding=True, dynamic_axes=None,
                  output_names=None, input_names=None):
        model.eval()

        onnx_io = io.BytesIO()
        # export to onnx with the first input
        torch.onnx.export(model, inputs_list[0], onnx_io,
                          do_constant_folding=do_constant_folding, opset_version=12,
                          dynamic_axes=dynamic_axes, input_names=input_names, output_names=output_names)
        # validate the exported model with onnx runtime
        for test_inputs in inputs_list:
            with torch.no_grad():
                if isinstance(test_inputs, torch.Tensor) or isinstance(test_inputs, list):
                    test_inputs = (nested_tensor_from_tensor_list(test_inputs),)
                test_ouputs = model(*test_inputs)
                if isinstance(test_ouputs, torch.Tensor):
                    test_ouputs = (test_ouputs,)
            self.ort_validate(onnx_io, test_inputs, test_ouputs, tolerate_small_mismatch)

    def ort_validate(self, onnx_io, inputs, outputs, tolerate_small_mismatch=False):

        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))

        ort_session = onnxruntime.InferenceSession(onnx_io.getvalue())
        # compute onnxruntime output prediction
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = ort_session.run(None, ort_inputs)
        for i, element in enumerate(outputs):
            try:
                torch.testing.assert_allclose(element, ort_outs[i], rtol=1e-03, atol=1e-05)
            except AssertionError as error:
                if tolerate_small_mismatch:
                    self.assertIn("(0.00%)", str(error), str(error))
                else:
                    raise

    def test_model_onnx_detection(self):
        model = detr_resnet50(pretrained=False).eval()
        dummy_image = torch.ones(1, 3, 800, 800) * 0.3
        model(dummy_image)

        # Test exported model on images of different size, or dummy input
        self.run_model(
            model,
            [(torch.rand(1, 3, 750, 800),)],
            input_names=["inputs"],
            output_names=["pred_logits", "pred_boxes"],
            tolerate_small_mismatch=True,
        )

    @unittest.skip("CI doesn't have enough memory")
    def test_model_onnx_detection_panoptic(self):
        model = detr_resnet50_panoptic(pretrained=False).eval()
        dummy_image = torch.ones(1, 3, 800, 800) * 0.3
        model(dummy_image)

        # Test exported model on images of different size, or dummy input
        self.run_model(
            model,
            [(torch.rand(1, 3, 750, 800),)],
            input_names=["inputs"],
            output_names=["pred_logits", "pred_boxes", "pred_masks"],
            tolerate_small_mismatch=True,
        )

import numpy as np
import torchvision
if __name__ == '__main__':
    # masks = np.array([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]])

    # ids = np.array([1,3,5])
    # masks = masks == ids[:, None, None]
    # print(ids[:, None, None].shape)
    # print(masks)

    # A = torch.rand((2*2,3,4))
    # B = torch.nn.functional.interpolate(A[:, None], size=(6,8),
    #                             mode="bilinear", align_corners=False)
    # print(B.shape)

    # C = torchvision.ops.misc.interpolate(A[:, None], size=(6,8),
    #                             mode="bilinear", align_corners=False)    
    # print(C.shape)


    # from scipy.optimize import linear_sum_assignment
    
    # cost =np.array([[4,1,3],[2,0,5],[3,2,2]])
    # row_ind,col_ind=linear_sum_assignment(cost)
    # print(row_ind)#开销矩阵对应的行索引
    # print(col_ind)#对应行索引的最优指派的列索引
    # print(cost[row_ind,col_ind])#提取每个行索引的最优指派列索引所在的元素，形成数组
    # print(cost[row_ind,col_ind].sum())#数组求和

    raw_logits = torch.rand((2,10,8))
    cur_logits = raw_logits[0]
    threshold = 0.1

    scores, labels = cur_logits.softmax(-1).max(-1)
    print(scores)
    print(labels)
    keep = labels.ne(raw_logits.shape[-1] - 1) & (scores > threshold)# raw_logits.shape[-1]-1 为背景分类
    print(keep)
    
    def get_ids_area(masks, scores, dedup=False):
        # This helper function creates the final panoptic segmentation image
        # It also returns the area of the masks that appears on the image

        m_id = masks.transpose(0, 1).softmax(-1)

        if m_id.shape[-1] == 0:
            # We didn't detect any mask :(
            m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
        else:
            m_id = m_id.argmax(-1).view(h, w)

        if dedup:
            # Merge the masks corresponding to the same stuff class
            for equiv in stuff_equiv_classes.values():
                if len(equiv) > 1:
                    for eq_id in equiv:
                        m_id.masked_fill_(m_id.eq(eq_id), equiv[0])

        final_h, final_w = to_tuple(target_size)

        seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
        seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

        np_seg_img = (
            torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
        )
        m_id = torch.from_numpy(rgb2id(np_seg_img))

        area = []
        for i in range(len(scores)):
            area.append(m_id.eq(i).sum().item())
        return area, seg_img

    # unittest.main()


