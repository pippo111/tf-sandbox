import numpy as np

""" Converts scan image to cuboid by padding with zeros
"""
def cubify_scan(data: np.ndarray, cube_dim: int) -> np.ndarray:
    pad_w = (cube_dim - data.shape[0]) // 2
    pad_h = (cube_dim - data.shape[1]) // 2
    pad_d = (cube_dim - data.shape[2]) // 2

    data = np.pad(
        data,
        [(pad_w, pad_w), (pad_h, pad_h), (pad_d, pad_d)],
        mode='constant',
        constant_values=0
    )
    
    return data
