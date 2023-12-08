'''
Handles global system and device settings
'''

import torch
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('gpu')
else:
    DEVICE = torch.device('cpu')

MACHINE_EPSILON = torch.finfo(float).eps