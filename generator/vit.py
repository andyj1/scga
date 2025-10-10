import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViTGenerator(nn.Module):
    """
    Vision Transformer based Generator for adversarial attacks.
    
    Args:
        image_size (int): The size of the input images (assumed to be square).
        patch_size (int): The size of patches to divide the image into.
        in_channels (int): Number of input image channels (3 for RGB).
        dim (int): The embedding dimension.
        depth (int): Number of transformer blocks.
        heads (int): Number of attention heads.
        mlp_dim (int): Dimension of the MLP in the transformer block.
        dim_head (int): Dimension of each attention head.
        dropout (float): Dropout rate.
    """
    def __init__(
        self, 
        image_size=224, 
        patch_size=16, 
        in_channels=3, 
        dim=768, 
        depth=12, 
        heads=12, 
        mlp_dim=3072, 
        dim_head=64, 
        dropout=0.0, 
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        
        # Image to patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, dim),
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # Patch to image projection (generates perturbation)
        self.to_perturbation = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_size * patch_size * in_channels),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                      p1=patch_size, p2=patch_size, h=image_size//patch_size, w=image_size//patch_size),
            nn.Tanh()  # Constrains output to [-1, 1]
        )
    
    def forward(self, img):
        # Convert image to patch embeddings
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        # Add positional embeddings
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Generate perturbation
        perturbation = self.to_perturbation(x)
        
        return perturbation

class MultiScaleViTGenerator(nn.Module):
    def __init__(self, image_size=224, patch_sizes=[16, 8], in_channels=3, 
                 dim=768, depth=12, heads=12, dim_head=64, mlp_dim=3072, dropout=0.0):
        super().__init__()
        
        
        # Create separate transformer branches for each patch size
        self.branches = nn.ModuleList()
        for patch_size in patch_sizes:
            num_patches = (image_size // patch_size) ** 2
            
            # Create a ModuleDict for this branch
            branch = nn.ModuleDict({
                'embedding': nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                    nn.Linear(patch_size * patch_size * in_channels, dim),
                ),
                'transformer': Transformer(dim, depth//2, heads, dim_head, mlp_dim, dropout),
                'to_perturbation': nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, patch_size * patch_size * in_channels),
                    Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
                              p1=patch_size, p2=patch_size, h=image_size//patch_size, w=image_size//patch_size),
                )
            })
            
            # Properly register the positional embedding as a parameter
            branch.register_parameter('pos_embedding', nn.Parameter(torch.randn(1, num_patches, dim)))
            
            self.branches.append(branch)
            
        # Final layer to combine perturbations from different scales
        self.combiner = nn.Conv2d(in_channels * len(patch_sizes), in_channels, kernel_size=1)
        self.final_activation = nn.Tanh()
    
    def forward(self, img):
        perturbations = []
    
        # Process image at each scale
        for branch in self.branches:
            # Convert image to patch embeddings
            x = branch['embedding'](img)
            b, n, _ = x.shape
            
            # Add positional embeddings - accessing as an attribute, not a dictionary key
            x = x + branch.pos_embedding[:, :n]  # Notice the change here
            
            # Apply transformer
            x = branch['transformer'](x)
            
            # Generate perturbation at this scale
            pert = branch['to_perturbation'](x)
            perturbations.append(pert)
        
        # Concatenate perturbations from different scales
        multi_scale_pert = torch.cat(perturbations, dim=1)
        
        # Combine perturbations
        perturbation = self.combiner(multi_scale_pert)
        perturbation = self.final_activation(perturbation)
        
        # perturbation = self.final_activation(perturbation) * self.epsilon
        
        # Add perturbation to original image
        # adversarial = torch.clamp(img + perturbation, 0, 1)
        
        return perturbation


class AttentionVisualization(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.attention_maps = []
        
        # Register hook to capture attention maps
        def hook_fn(module, input, output):
            self.attention_maps.append(output.detach())
        
        # Register hooks on attention modules
        for name, module in self.generator.named_modules():
            if isinstance(module, Attention):
                module.register_forward_hook(hook_fn)
    
    def forward(self, x):
        self.attention_maps = []  # Clear previous maps
        result = self.generator(x)
        return result, self.attention_maps
    
    def visualize_attention(self, image_idx=0, head_idx=0, save_path=None):
        """Visualize attention maps for a specific image and attention head."""
        import matplotlib.pyplot as plt
        
        # Get dimensions
        num_layers = len(self.attention_maps)
        fig, axes = plt.subplots(1, num_layers, figsize=(num_layers * 4, 4))
        
        for i, attn_map in enumerate(self.attention_maps):
            # Extract attention weights for the specified image and head
            # Shape: [batch, heads, seq_len, seq_len]
            weights = attn_map[image_idx, head_idx].cpu().numpy()
            
            # Plot heatmap
            im = axes[i].imshow(weights, cmap='viridis')
            axes[i].set_title(f"Layer {i+1}")
            fig.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

if __name__ == '__main__':
    # Test the ViTGenerator
    import time 
    netG = ViTGenerator()
    img = torch.randn(1, 3, 224, 224)
    start = time.time()
    perturbation = netG(img)
    end = time.time()
    print(perturbation.shape)  # Expected: (1, 3, 224, 224)
    print(f'Elapsed time: {end - start:.4f} seconds')
    
    
    # Test the ViTGenerator
    import time 
    netG = MultiScaleViTGenerator()
    img = torch.randn(1, 3, 224, 224)
    start = time.time()
    perturbation = netG(img)
    end = time.time()
    print(perturbation.shape)  # Expected: (1, 3, 224, 224)
    print(f'Elapsed time: {end - start:.4f} seconds')
    