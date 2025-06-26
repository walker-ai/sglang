from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from sgl_kernel import sage_ops
except:
    raise ImportError("Can not import sgl_kernel. Please check your installation.")

