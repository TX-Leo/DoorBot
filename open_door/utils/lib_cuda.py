import torch
print(torch.cuda.is_available())

import os
cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
print(cuda_home)

print(torch.cuda._is_compiled())
CUDA_HOME = _find_cuda_home() if torch.cuda._is_compiled() else None
print(CUDA_HOME)