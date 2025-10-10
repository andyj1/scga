import os
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np

def save_annotated_images(image_root, json_file_path):
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)
    class_names = coco_data['categories']
    class_id_to_name = {cat['id']: cat['name'] for cat in class_names}
    output_dir = os.path.join(os.path.dirname(json_file_path), 'visualization')
    os.makedirs(output_dir, exist_ok=True)

    for image_info in tqdm(coco_data['images']):
        image_id = image_info['id']
        image_path = os.path.join(image_root, image_info['file_name'])
        image_name = image_info['file_name'].split('/')[-1]
        img = Image.open(image_path)
        image_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]

        fig, ax = plt.subplots(1)
        ax.imshow(img)
        ax.set_xlim(0, img.width)
        ax.set_ylim(img.height, 0)
        ax.axis('off')
        assigned_colors = [np.random.rand(3) for _ in range(len(image_annotations))]
        for i, anno in enumerate(image_annotations):
            bbox = anno['bbox']  # [x, y, width, height]
            class_id = anno['category_id']
            class_name = class_id_to_name[class_id]

            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1.5, edgecolor=assigned_colors[i], facecolor='none'
            )
            ax.add_patch(rect)

            plt.text(
                bbox[0], bbox[1]-7, class_name, color=assigned_colors[i], fontsize=8,
                bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.1')
            )
            
        output_image_path = os.path.join(output_dir, image_name)
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str, help='Root directory where images are located.')
    parser.add_argument('--json_file_path', type=str, help='Path to the JSON annotation file.')
    args = parser.parse_args()
    
    save_annotated_images(args.image_root, args.json_file_path)