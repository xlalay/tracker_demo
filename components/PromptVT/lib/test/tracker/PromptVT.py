from ...test.tracker.basetracker import BaseTracker
import torch
from ...train.data.processing_utils import sample_target
import cv2
import os
from ...utils.box_ops import clip_box
from ...test.tracker.utils import PreprocessorX_onnx
import multiprocessing
import numpy as np
import time
from ...models.PromptVT.mobileone import  reparameterize_model
from ...models.PromptVT import build_PromptVT
from ...test.tracker.utils import PreprocessorX
from ...utils.merge import get_qkv
from ...test.parameter.PromptVT import parameters

class PromptVT(BaseTracker):
    def __init__(self, params, dataset_name, model_path=None):
        super(PromptVT, self).__init__(params)
        network = build_PromptVT(params.cfg, phase='test')
        missing_keys, unexpected_keys = network.load_state_dict(torch.load(model_path, map_location='cpu')['net'], strict=True)
        # network.load_state_dict(torch.load(model_path, map_location='cpu')['net'], strict=True)
        network.deep_sup = False  # disable deep supervision during the test stage
        network.distill = False  # disable distillation during the test stage
        print("Testing Dataset: ",dataset_name)
        self.cfg = params.cfg
        self.network = network.cuda() #for gpu
        # self.network = network
        self.network.eval()
        self.network = reparameterize_model(self.network)
        self.preprocessor = PreprocessorX() #for gpu
        # self.preprocessor = PreprocessorX_cpu()
        self.state = None
        self.z_dict1 = {}
        self.z_dict2 = {}
        self.z_dict_list = []
        # for debug
        self.debug = False
        self.frame_id = 0
        self.flag = False
        self.conf=0
        self.interval=0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        self.z_dict1 = {}
        self.Datasetname = dataset_name
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
        self.num_extra_template = len(self.update_intervals)
        self.model_time_2 = 0

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], 2,
                                                    128)
        x_patch_arr_s, _, x_amask_arr_s = sample_target(image, info['init_bbox'], 2,
                                                    128)
        template1, template_mask1 = self.preprocessor.process(z_patch_arr, z_amask_arr)
        template2, template_mask2 = self.preprocessor.process(x_patch_arr_s, x_amask_arr_s)
        with torch.no_grad():
            self.z_dict1 = self.network.forward_backbone(template1, zx="template0", mask=template_mask1)
            self.z_dict2 = self.network.forward_backbone(template2, zx="template0", mask=template_mask2)
        self.z_dict_list.append(self.z_dict1)
        self.z_dict_list.append(self.z_dict2)
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.Datasetname == 'lasot':
            self.conf = 0.4
            self.interval = 70
        elif self.Datasetname == 'uav':
            self.conf = 0.5
            self.interval = 100
        elif self.Datasetname == 'got10k_test':
            self.conf = 0.6
            self.interval = 30
        elif self.Datasetname == 'trackingnet':
            self.conf = 0.6
            self.interval = 20
        elif self.Datasetname == 'vot20':
            self.conf = 0.5
            self.interval = 10
        elif self.Datasetname == 'otb':
            self.conf = 0.6
            self.interval = 20
        elif self.Datasetname == 'video':
            self.conf = 0.5
            self.interval = 30
        else:
            raise ValueError("Invalid dataset.")

    def track(self, image, info: dict = None,intervals =None,search_factor = None,conf = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, search_factor,
                                                                320)  # (x1, y1, w, h)
        search, search_mask = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            model_time_start = time.time()
            x_dict = self.network.forward_backbone(search, zx="search", mask=search_mask)
            feat_dict_list = [self.z_dict_list[0], self.z_dict_list[1], x_dict]
            src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16,dy_temp_8,dy_temp_16 = get_qkv(feat_dict_list)
            
            out_dict, _, _ = self.network.forward_transformer(src_temp_8 = src_temp_8  , pos_temp_8 = pos_temp_8, src_temp_16 = src_temp_16,
                                                              pos_temp_16 = pos_temp_16, src_search_8 = src_search_8, pos_search_8 = pos_search_8 ,
                                                              src_search_16 = src_search_16 ,pos_search_16 = pos_search_16,dy_temp_8 = dy_temp_8,dy_temp_16 = dy_temp_16,run_box_head=True, run_cls_head=True,
                                                              flag = self.flag , framid=self.frame_id )
            model_time_1 = time.time() - model_time_start
        self.flag = False
        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        conf_score = out_dict["pred_logits"].view(-1).sigmoid().item()
    
        
        for idx, update_i in enumerate(self.update_intervals):
            if self.frame_id % self.interval == 0 and conf_score > self.conf:
                z_patch_arr, _, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                            output_sz=self.params.template_size)  # (x1, y1, w, h)
                template_t,template_mask = self.preprocessor.process(z_patch_arr, z_amask_arr)
                with torch.no_grad():
                    model_time_start = time.time()
                    z_dict_t = self.network.forward_backbone(template_t, zx="template0", mask=template_mask)

                    self.model_time_2 = time.time() - model_time_start
                self.z_dict_list[idx+1] = z_dict_t  # the 1st element of z_dict_list is template from the 1st frame
                self.flag = True

        if(self.flag ==True):
            model_time = model_time_1 + self.model_time_2
        else:
            model_time = model_time_1

        # if self.debug:
        #     x1, y1, w, h = self.state
        #     image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #     cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
        #     save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
        #     cv2.imwrite(save_path, image_BGR)
        return {"target_bbox": self.state,
                "model_time": model_time,
                "conf": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]