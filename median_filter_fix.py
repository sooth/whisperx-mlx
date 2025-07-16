"""Fixed median filter for 2D arrays."""

import numpy as np
from scipy import signal

def median_filter_fixed(x: np.ndarray, filter_width: int) -> np.ndarray:
    """Apply a median filter of width `filter_width` along the last dimension of `x`"""
    pad_width = filter_width // 2
    if x.shape[-1] <= pad_width:
        return x
    
    assert filter_width > 0 and filter_width % 2 == 1, "`filter_width` should be an odd number"
    
    # For 2D input, work directly
    if x.ndim == 2:
        # Pad and filter each row
        result = np.zeros_like(x)
        for i in range(x.shape[0]):
            padded = np.pad(x[i], pad_width, mode='reflect')
            result[i] = signal.medfilt(padded.astype(np.float32), kernel_size=filter_width)[pad_width:-pad_width]
        return result
    else:
        # Original logic for other dimensions
        if x.ndim <= 2:
            x = x[None, None, :]
        
        x = np.pad(x, ((0, 0), (0, 0), (pad_width, pad_width)), mode="reflect")
        result = signal.medfilt(x.astype(np.float32), kernel_size=(1, 1, filter_width))[
            ..., pad_width:-pad_width
        ]
        
        if x.ndim <= 2:
            result = result[0, 0]
            
        return result