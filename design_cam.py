import torch
import os
from datetime import datetime
import numpy as np
from models import *
from tqdm import tqdm
from skimage import io
from skimage import img_as_ubyte
import torch
import matplotlib.pyplot as plt
from evaluation import *
from utils import * 

class Design_CAM(object):

    def __init__(self, args): 

        self.project_path = args.project_path
        self.record_path = args.record_path
        self.task = args.task  # 'binary' or 'multiclass'
        
        multi_model = Res18_Classifier(num_classes=args.num_classes)
        binary_model = Res18_Classifier(num_classes=1)
        bin_score_model = Res_Scoring().cuda()
        multi_score_model=Res_Scoring().cuda()
        binary_model.load_pretrain_weight(args.bin_pretrained_path)
        multi_model.load_pretrain_weight(args.multi_pretrained_path)
        bin_score_model.load_pretrain_weight(args.bin_score_model_pretrained_path)
        multi_score_model.load_pretrain_weight(args.multi_score_model_pretrained_path)

        for param in binary_model.parameters():
            param.requires_grad = False
        for param in multi_model.parameters():
            param.requires_grad = False
        for param in bin_score_model.parameters():
            param.requires_grad = False
        for param in multi_score_model.parameters():
            param.requires_grad = False

        if len(args.gpu_ids) > 1:
            binary_model = torch.nn.DataParallel(binary_model, device_ids=args.gpu_ids)
            multi_model = torch.nn.DataParallel(multi_model, device_ids=args.gpu_ids)
            bin_score_model = torch.nn.DataParallel(bin_score_model, device_ids=args.gpu_ids)
            multi_score_model = torch.nn.DataParallel(multi_score_model, device_ids=args.gpu_ids)

        self.binary_model = binary_model.to('cuda').eval()
        self.multi_model = multi_model.to('cuda').eval()
        self.bin_score_model = bin_score_model.to('cuda').eval()
        self.multi_score_model = multi_score_model.to('cuda').eval()

        # Define save directory
        self.save_dir = os.path.join(self.project_path, self.record_path, f"{self.task}_eval")
        os.makedirs(self.save_dir, exist_ok=True)
    
    def step(self, img):

        img = img.cuda()

        # --- Binary and Multi CAM extraction ---
        logits_collect_binary, map_collect_binary = self.binary_model(img)
        logits_collect_multi, map_collect_multi = self.multi_model(img)
        
        # --- Binary ame map computation ---
        map_collect_binary_copy = [t.clone() for t in map_collect_binary]
        _, _, _, bin_ame_map = self.bin_score_model(img, map_collect_binary_copy)
        bin_ame_map = self.normalize_map(bin_ame_map)

        # --- binary guidance ---
        map_collect_multi = torch.stack(map_collect_multi, dim=0)
        map_collect = bin_ame_map * map_collect_multi

        # --- Unbind (convert to list of tensors) ---
        map_collect = list(map_collect.unbind(0))
        
        # --- Multi-class score model ---
        _, _, _, ame_map = self.multi_score_model(img, map_collect)

        # --- Final attention maps ---
        ame_map = torch.cat((ame_map, bin_ame_map), dim=1)
        
        return ame_map.detach().cpu(), logits_collect_binary[-1].detach().cpu(), logits_collect_multi[-1].detach().cpu()
    
    def normalize_map(self, att_map: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        n, c, h, w = att_map.size()
        flat = att_map.view(n, c, -1)
        min_val = flat.min(2, keepdim=True)[0]
        max_val = flat.max(2, keepdim=True)[0]
        normalized = (flat - min_val) / (max_val - min_val + eps)
        return normalized.view(n, c, h, w)


    def postprocess_cam(self, cam, binary_cam, thresholds=None):

        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        binary_cam = np.maximum(binary_cam, 0)
        binary_cam = binary_cam / binary_cam.max()

        cam =np.where(binary_cam>thresholds, cam*binary_cam, 0)
        cam = np.where(cam>0.4, 1, 0)

        return cam

    def run_tumor_test(self, loader, threshold=None):
        self.binary_model.eval()
        self.multi_model.eval()
        self.bin_score_model.eval()
        self.multi_score_model.eval()
        
        log_path = os.path.join(self.save_dir, "results.log")
        log_file = open(log_path, "w+")
        log_file.writelines(str(datetime.now())+"\n")
        log_file.close()
        csv_path = os.path.join(self.save_dir, "tumor_result.csv")
        csv_file = open(csv_path, "w+")
        csv_file.writelines("Img Name, Core Dice, Core IoU, Core HD95, Edema Dice, Edema IoU, Edema HD95\n")
        csv_file.close()

        test_bar = tqdm(loader)

        result_metric = {
            'Core Dice': [],
            'Core IoU': [],
            'Core HD95': [],
            'Edema Dice': [],
            'Edema IoU': [],
            'Edema HD95': []
        }

        with torch.no_grad():
            for img_name, case_batch, seg_batch in test_bar:
                img_name = img_name[0][:-4]

                ame_map, binary_logit, class_logit = self.step(case_batch)

                binary_logit = binary_logit.squeeze(0).cpu().numpy()
                class_logit = class_logit.squeeze(0).cpu().numpy()

                logit = np.concatenate((class_logit, binary_logit), axis=0)
                input_image = case_batch[0].permute(1, 2, 0)
                ame_map = self.CAM_algo(input_image, ame_map, img_name)

                results={}

                for i, class_name in enumerate(['core', 'edema']):
                    if class_name == 'core':
                        gt = np.where(seg_batch[0][0].numpy()!=0, 1, 0) + np.where(seg_batch[0][1].numpy()!=0, 1, 0)
                    elif class_name == 'edema':
                        gt = np.where(seg_batch[0][2].numpy()!=0, 1, 0)
                    else:
                        raise ValueError(f"Unknown class name: {class_name}")
                    
                    final_seg = self.postprocess_cam(ame_map[i], ame_map[-1], threshold)
                    
                    #using predicted logit to determine no tumor or tumor
                    if logit[i] < 0.5:
                        final_seg = np.zeros_like(gt)
                    result = compute_seg_metrics(gt, final_seg)
                    results[class_name] = result

                csv_file = open(csv_path, "a")
                csv_file.writelines(f"{img_name}, {results['core']['Dice']:.3f}, {results['core']['IoU']:.3f}, {results['core']['HD95']:.3f}, {results['edema']['Dice']:.3f}, {results['edema']['IoU']:.3f}, {results['edema']['HD95']:.3f}\n")
                csv_file.close()

                for k, v in results.items():
                    result_metric[f"{k.capitalize()} Dice"].append(v['Dice'])
                    result_metric[f"{k.capitalize()} IoU"].append(v['IoU'])
                    result_metric[f"{k.capitalize()} HD95"].append(v['HD95'])

        for k, v in result_metric.items():
            result_metric[k] = np.mean(v)
        test_bar.close()

        log_file = open(log_path, "a")
        log_file.writelines("Average Results\n")
        for k, v in result_metric.items():
            log_file.writelines(f"{k}: {v:.3f}\n")
        log_file.close()

        return result_metric

    def CAM_algo(self, input_image, ame_map, img_name, output_hist=False):
        
        for i in range(ame_map.shape[1]):
            if (ame_map[0][i].max() - ame_map[0][i].min()) > 0:
                ame_map[0][i] = (ame_map[0][i] - ame_map[0][i].min()) / (ame_map[0][i].max() - ame_map[0][i].min()+1e-5)

        ame_map = ame_map.squeeze(0).numpy()
        ame_map  = (1-ame_map)
        return ame_map
    
        