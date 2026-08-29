import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch.nn as nn

class wav2vec2(nn.Module):
    def __init__(self, target_layers, device = "cuda", model_name = "facebook/wav2vec2-base-100h"):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, device=device)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad =False

        self.target_layers = target_layers
        self.device=device

    def forward (self, input_values):
        with torch.no_grad():
            output = self.model.forward(input_values, output_hidden_states=True)
            all_layers = output.hidden_states
            sellected_features = [all_layers[i] for i in self.target_layers]
            if len(sellected_features) == 1:
                combined_features = sellected_features[0]
            else:
                combined_features = torch.stack(sellected_features, dim=0).mean(dim=0)

        return combined_features