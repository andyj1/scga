import torch
import numpy as np
from sklearn.cluster import KMeans
from resnet import GeneratorResnet
from dataloader import get_dataloader
from feature_extractor import FeatureExtractor
import argparse
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from rich import print

def calculate_mean_iou(pred_mask, gt_mask):
    """
    Calculate the mean Intersection over Union (IoU) between predicted clusters and ground truth mask.
    
    This function computes the maximum IoU across all clusters in the predicted mask when compared
    to the ground truth object mask. It iterates through each cluster in the prediction and calculates
    the IoU with the ground truth object (pixels labeled as 1), returning the highest IoU value found.
    
    Args:
        pred_mask (numpy.ndarray): Predicted segmentation mask with cluster labels (0, 1, 2, ...)
        gt_mask (numpy.ndarray): Ground truth binary mask where 1 indicates object pixels
    
    Returns:
        float: Maximum IoU value across all clusters (range: 0.0 to 1.0)
    """
    max_iou = 0.0
    num_clusters = np.max(pred_mask) + 1
    for i in range(num_clusters):
        pred_i = (pred_mask == i)
        gt_object = (gt_mask == 1)
        intersection = np.logical_and(pred_i, gt_object).sum()
        union = np.logical_or(pred_i, gt_object).sum()
        iou = intersection / union if union > 0 else 0.0
        if iou > max_iou:
            max_iou = iou
    return max_iou

def get_color_map(num_clusters):
    colors = [
        (100,100,100, 80), # Gray
        # (255, 56, 56, 80),   # Red
        (56, 255, 56, 80),   # Green
        # (56, 56, 255, 80),   # Blue
        # (255, 255, 56, 80),  # Yellow
        # (56, 255, 255, 80),  # Cyan
        # (255, 56, 255, 80),  # Magenta
        # (255, 255, 255, 80), # White
    ]
    return colors[:num_clusters]

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneratorResnet().to(device)

    # PRETRAINED_MODEL_PATH = '/mnt/sdb/advattack/BIA_working/from_50100/gmcv2025_exp0423/PDCL_OCCL_v16.0.1_BIA_vgg16_CLIPA_ViTB16_PG_contrastive_lamb+1_seed+1_GAMA_only/teacher_BIA+CLIPA_PG_0.pth'
    
    # CDA
    # PRETRAINED_MODEL_PATH = '/mnt/sdf/datasets/trained_attack_models/CDA/netG_-1_img_vgg16_imagenet_0_rl.pth'
    # LTP
    # PRETRAINED_MODEL_PATH = '/mnt/sdf/datasets/trained_attack_models/LTPexp0308_yhm/vgg16_LTP_2_scratch_normal_rho+0.01_sigma+8.0_beta1+0.5_batch+16_lr+0.0002_seed+0_freq+7_freq2+112_low_high_rand_CL_beta+25/netG_BIA_0.pth'
    # BIA
    # PRETRAINED_MODEL_PATH = '/mnt/sdf/datasets/trained_attack_models/BIA/vgg16/netG_BIA_0.pth'
    # BIA + SSGA
    # PRETRAINED_MODEL_PATH = '/mnt/sdb/advattack/BIA_working/experiments_1/gmcv_final_250423_T174816_0421_BIA_MT_featuredistillation_blocks01_tau0.6_seed_2_resnet_vgg16_/0421_BIA_MT_featuredistillation_blocks01_tau0.6_seed_2_resnet_vgg16/teacher_0421_BIA_MT_featuredistillation_blocks01_tau0.6_seed_2_0.pth'
    # GAMA
    # PRETRAINED_MODEL_PATH = '/mnt/sdf/datasets/trained_attack_models/GAMA/exp0921/BIA+GAMA_vgg16/netG_BIA+GAMA_0.pth'
    # FACL
    # PRETRAINED_MODEL_PATH = '/mnt/sdf/datasets/trained_attack_models/FACL/exp0226_final/vgg16_BIA+FA+CL_scratch_normal_rho+0.01_sigma+8.0_beta1+0.5_batch+16_lr+0.0002_seed+0_freq+7_freq2+112_low_high_rand_CL_beta+25/netG_BIA+FA_0.pth'
    # PDCL
    # PRETRAINED_MODEL_PATH = '/mnt/sdf/datasets/trained_attack_models/PDCL/exp1121_final/final_v16.0.1_BIA_vgg16_CLIPA_ViTB16_PG_contrastive_lamb+1_seed+1_final/netG_BIA+CLIPA_PG_0.pth'
    
    
    # CDA Ours
    # PRETRAINED_MODEL_PATH='/mnt/sdb/advattack/BIA_working/baselines/CDA/saved_models_neurips_seed_1/teacher_imdep_vgg16.pth'
    # LTP Ours
    # PRETRAINED_MODEL_PATH='/mnt/sdb/advattack/BIA_working/from_5236/250514_T001727_LTP_MT_featdistillation_resnet_vgg16/teacher_BIA_0.pth'
    # # BIA Ours
    # PRETRAINED_MODEL_PATH='/mnt/sdb/advattack/BIA_working/experiments_1/gmcv_final_250423_T174816_0421_BIA_MT_featuredistillation_blocks01_tau0.6_seed_2_resnet_vgg16_/0421_BIA_MT_featuredistillation_blocks01_tau0.6_seed_2_resnet_vgg16/teacher_0421_BIA_MT_featuredistillation_blocks01_tau0.6_seed_2_0.pth'
    # # GAMA Ours
    PRETRAINED_MODEL_PATH='/mnt/sdb/advattack/BIA_working/from_50100/gmcv2025_exp0423/GAMA_vgg16_seed+0/teacher_BIA+GAMA_0.pth'
    # # FACL Ours
    # PRETRAINED_MODEL_PATH='/mnt/sdb/advattack/BIA_working/baselines/FACL_Attack/logs/neurips2025_04232025/FACL_vgg16_low_high_rand_seed+0/teacher_BIA+FA_0.pth'
    # # PDCL Ours
    # PRETRAINED_MODEL_PATH='/mnt/sdb/advattack/BIA_working/from_50100/gmcv2025_exp0423/PDCL_OCCL_v16.0.1_BIA_vgg16_CLIPA_ViTB16_PG_contrastive_lamb+1_seed+1_GAMA_only/teacher_BIA+CLIPA_PG_0.pth'
    
    
    
    ckpt = torch.load(PRETRAINED_MODEL_PATH, map_location=torch.device('cpu'))
    if 'model_state_dict'in ckpt: ckpt = ckpt['model_state_dict']
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ckpt.items()})
    model = model.cuda().eval()
    
    layers_to_analyze = ['resblock1', 'resblock2', 'resblock3', 'resblock4', 'resblock5', 'resblock6']
    # layers_to_analyze = ['resblock4', 'resblock5', 'resblock6']
    color_map = get_color_map(args.num_clusters)
    
    print(f"Analyzing layers: {layers_to_analyze}")
    dataloader = get_dataloader(args.data_root, split=args.split, subset=args.subset, batch_size=args.batch_size)

    # Keep track of which ground truth images have been saved to avoid duplication                                                                               │
    saved_gt_paths = set()

    total_iou = np.zeros(len(layers_to_analyze))
    image_count = 0
    
    progress_bar = tqdm(dataloader, desc=f"IoU for feature k-means clusters", leave=False, ncols=100, position=0,total=len(dataloader))
    with torch.no_grad():
        for images, gt_masks, img_paths, class_names in progress_bar:
            images = images.to(device)
            x, feature_maps = model(images, feat=True)
            # feature_maps = feature_extractor(images)

            for i in range(x.size(0)):
                image_count += 1
                for idx, layer_name in enumerate(layers_to_analyze):
                    progress_bar.set_description(f"IoU for feature k-means clusters: {image_count}/{len(dataloader)} for layer {layer_name}")
                    feature_map = feature_maps[layer_name][i]
                    gt_mask = gt_masks[i]
                    img_path = img_paths[i]
                    class_name = class_names[i]

                    h, w = feature_map.shape[1], feature_map.shape[2]
                    gt_mask_resized = torch.nn.functional.interpolate(gt_mask.unsqueeze(0), size=(h, w), mode='nearest').squeeze(0)
                    
                    unique_labels = torch.unique(gt_mask_resized)
                    object_label = unique_labels[-1] if len(unique_labels) > 1 else 1
                    binary_gt_mask = (gt_mask_resized == object_label).cpu().numpy().astype(int)

                    pixels = feature_map.permute(1, 2, 0).reshape(-1, feature_map.size(0)).cpu().numpy()
                    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0, n_init=10)
                    predicted_labels = kmeans.fit_predict(pixels)
                    predicted_mask = predicted_labels.reshape(h, w)
                    
                    iou = calculate_mean_iou(predicted_mask, binary_gt_mask.squeeze())
                    total_iou[idx] += iou
                    
                    if args.save_visuals:
                        original_image = Image.open(img_path).convert("RGBA")
                        image_basename = os.path.splitext(os.path.basename(img_path))[0]                                                                         
                        output_class_dir = os.path.join(args.output_dir, args.split, class_name)                                                                 
                        os.makedirs(output_class_dir, exist_ok=True)   
                        
                        overlay_np = np.zeros((h, w, 4), dtype=np.uint8)
                        for cluster_id, color in enumerate(color_map):
                            overlay_np[predicted_mask == cluster_id] = color
                        overlay_image = Image.fromarray(overlay_np, 'RGBA')
                        overlay_resized = overlay_image.resize(original_image.size, Image.NEAREST)
                        
                        final_image = Image.alpha_composite(original_image, overlay_resized)
                        draw = ImageDraw.Draw(final_image)
                        try:
                            font = ImageFont.truetype("arial.ttf", 24)
                        except IOError:
                            font = ImageFont.load_default()
                        # draw.text((10, 10), f"Layer: {layer_name}\nIoU: {iou:.4f}", fill="white", font=font)
                        
                        # image_basename = os.path.splitext(os.path.basename(img_path))[0]
                        # output_dir = os.path.join(args.output_dir, args.split, image_basename)
                        # os.makedirs(output_dir, exist_ok=True)
                        # output_path = os.path.join(output_dir, f"{layer_name}_clusters.png")
                        # final_image.convert('RGB').save(output_path)

                        output_path_pred = os.path.join(output_class_dir, f"{image_basename}_{layer_name}_clusters.png")                                         
                        final_image.convert('RGB').save(output_path_pred)                                                                                   
                                                                                                                                                                    
                        # --- Save Ground Truth Visualization (only once per image) ---                                                                          
                        if img_path not in saved_gt_paths:                                                                                                       
                            gt_mask_for_viz = (gt_mask_resized == object_label).cpu().numpy().squeeze()                                                          
                            overlay_gt = np.zeros((h, w, 4), dtype=np.uint8)                                                                                     
                            overlay_gt[gt_mask_for_viz == 1] = (56, 255, 56, 150) # Green overlay for GT                                                         
                            overlay_image_gt = Image.fromarray(overlay_gt, 'RGBA').resize(original_image.size, Image.NEAREST)                                    
                                                                                                                                                                    
                            final_image_gt = Image.alpha_composite(original_image, overlay_image_gt)                                                             
                            output_path_gt = os.path.join(output_class_dir, f"{image_basename}_ground_truth.png")                                                
                            final_image_gt.convert('RGB').save(output_path_gt)                                                                                   
                            saved_gt_paths.add(img_path)   
                        
        mean_iou = total_iou / image_count
        for idx, layer_name in enumerate(layers_to_analyze):
            print(f"Result for layer '{layer_name}': Mean IoU = {mean_iou[idx]:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Layer-wise K-Means analysis on ResNet Generator features.')
    parser.add_argument('--data_root', type=str, default='ImageNet-S/datapreparation/imagenet_s', help='Root directory of the ImageNet-S dataset.')
    parser.add_argument('--split', type=str, default='validation', choices=['train', 'test', 'validation'], help="Dataset split to use.")
    parser.add_argument('--subset', type=str, default='im50', choices=['im50', 'im300', 'all'], help="ImageNet-S subset to use.")
    parser.add_argument('--output_dir', type=str, default='results_visuals', help='Directory to save visual results.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. Use a small value due to per-image processing.')
    parser.add_argument('--num_clusters', type=int, default=3, help='Number of clusters for K-Means.')
    parser.add_argument('--save_visuals', action='store_true', default=False, help='Save images with overlaid cluster masks.')
    
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, f'{args.num_clusters}_clusters')
    main(args)