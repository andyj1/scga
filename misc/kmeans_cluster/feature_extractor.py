import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_name):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.feature = None
        
        # Register a forward hook to the desired layer
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self.hook)
                break
        else:
            raise ValueError(f"Layer {layer_name} not found in model.")

    def hook(self, module, input, output):
        self.feature = output

    def forward(self, x):
        # We don't need the model's final output, just the feature from the hook
        self.model(x)
        return self.feature
