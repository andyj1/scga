import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import re

def create_image_grids():
    """
    For each directory in validation folder, group images by their base name
    and create grids combining ground_truth and resblock cluster images.
    """
    
    # validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_bia/2_clusters/validation"
    # output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_bia/2_clusters/grids"
    
    # validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_bia_ours/2_clusters/validation"
    # output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_bia_ours/2_clusters/grids"
    
    # validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_pdcl/2_clusters/validation"
    # output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_pdcl/2_clusters/grids"
    
    # validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_gama/2_clusters/validation"
    # output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_gama/2_clusters/grids"
    
    # validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_ltp/2_clusters/validation"
    # output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_ltp/2_clusters/grids"
    
    # validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_cda/2_clusters/validation"
    # output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_cda/2_clusters/grids"
    
    
    
    
    
    validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_cda_ours/2_clusters/validation"
    output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_cda_ours/2_clusters/grids"
    
    validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_ltp_ours/2_clusters/validation"
    output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_ltp_ours/2_clusters/grids"
    
    validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_bia_ours/2_clusters/validation"
    output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_bia_ours/2_clusters/grids"
    
    validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_gama_ours/2_clusters/validation"
    output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_gama_ours/2_clusters/grids"
    
    validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_facl_ours/2_clusters/validation"
    output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_facl_ours/2_clusters/grids"
    
    validation_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_pdcl_ours/2_clusters/validation"
    output_dir = "/mnt/sdb/advattack/BIA_working/k_means_extractor/results_visuals_pdcl_ours/2_clusters/grids"
    
    
    
    
    
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the image suffixes we want to include
    suffixes = [
        'ground_truth.png',
        'resblock1_clusters.png',
        'resblock2_clusters.png', 
        'resblock3_clusters.png',
        'resblock4_clusters.png',
        'resblock5_clusters.png',
        'resblock6_clusters.png'
    ]
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(validation_dir) 
               if os.path.isdir(os.path.join(validation_dir, d))]
    
    for subdir in subdirs:
        print(f"Processing directory: {subdir}")
        subdir_path = os.path.join(validation_dir, subdir)
        
        # Get all PNG files in the subdirectory
        png_files = glob.glob(os.path.join(subdir_path, "*.png"))
        
        # Group files by their base name (everything before the suffix)
        image_groups = {}
        
        for png_file in png_files:
            filename = os.path.basename(png_file)
            
            # Check if this file ends with one of our target suffixes
            for suffix in suffixes:
                if filename.endswith(suffix):
                    # Extract the base name (everything before the suffix)
                    base_name = filename[:-len(suffix)]
                    
                    if base_name not in image_groups:
                        image_groups[base_name] = {}
                    
                    image_groups[base_name][suffix] = png_file
                    break
        
        # Create grids for each group
        for base_name, image_dict in image_groups.items():
            # Check if we have all 7 required images
            if len(image_dict) == 7:
                print(f"  Creating grid for: {base_name}")
                
                # Load images in the correct order
                images = []
                for suffix in suffixes:
                    if suffix in image_dict:
                        # Load image and convert to tensor
                        img = Image.open(image_dict[suffix]).convert('RGB')
                        transform = transforms.ToTensor()
                        img_tensor = transform(img)
                        images.append(img_tensor)
                    else:
                        print(f"    Warning: Missing {suffix} for {base_name}")
                
                if len(images) == 7:
                    # Use make_grid to create a proper grid layout
                    # First row: ground truth (centered)
                    # Second row: 6 resblock images
                    
                    # Create a 2x6 grid by arranging images properly
                    # We'll create a list of 12 images (2 rows of 6)
                    grid_images = []
                    
                    # First row: ground truth repeated 6 times (or centered with padding)
                    ground_truth = images[0]
                    resblock_images = images[1:]
                    
                    # Get dimensions
                    gt_h, gt_w = ground_truth.shape[1], ground_truth.shape[2]
                    rb_h, rb_w = resblock_images[0].shape[1], resblock_images[0].shape[2]
                    
                    # Resize ground truth to match resblock dimensions
                    ground_truth_resized = torch.nn.functional.interpolate(
                        ground_truth.unsqueeze(0), 
                        size=(rb_h, rb_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                    
                    # Create first row: ground truth in first position, then 5 empty/black images
                    first_row = [ground_truth_resized]
                    # Add 5 black images to complete the first row
                    for _ in range(5):
                        black_img = torch.zeros_like(ground_truth_resized)
                        first_row.append(black_img)
                    
                    # Second row: all 6 resblock images
                    second_row = resblock_images
                    
                    # Combine both rows
                    all_images = first_row + second_row
                    
                    # Create the grid using make_grid
                    grid = make_grid(all_images, nrow=6, padding=2, normalize=True)
                    
                    # Save the grid
                    output_filename = f"{subdir}_{base_name}_grid.png"
                    output_path = os.path.join(output_dir, output_filename)
                    save_image(grid, output_path)
                    
                    print(f"    Saved: {output_filename}")
                else:
                    print(f"    Skipping {base_name}: incomplete set")
            else:
                print(f"  Skipping {base_name}: found {len(image_dict)} images, need 7")

if __name__ == "__main__":
    create_image_grids()
    print("Grid creation completed!")
