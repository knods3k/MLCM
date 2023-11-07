'''
Handles global system and device settings
'''

import torch
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('gpu')
elif torch.cpu.is_available():
    DEVICE = torch.device('cpu')

