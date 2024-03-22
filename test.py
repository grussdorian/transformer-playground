import torch
from torch.nn import functional as F
import torch.nn as nn
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
    print("Running on MPS Device")
else:
    print("MPS device not found")

from multi_headed_self_attention import BigramLanguageModel
model = BigramLanguageModel()
model.load_state_dict(torch.load('./saved_models/multiheaded_attention.pth'))

model = model.to(mps_device)
model.inference(n_tokens=500, device=mps_device)
