import numpy as np

# Input image (5x5) and filter (3x3) stored as NumPy arrays
inp = np.array([[1,1,1,0,0],
                [0,1,1,1,0],
                [0,0,1,1,1],
                [0,0,1,1,0],
                [0,1,1,0,0]])
filt = np.array([[1,0,1],
                 [0,1,0],
                 [1,0,1]])

def conv2d(x, k, stride=1, pad=0):
    """2-D convolution (cross-correlation, as used in CNNs) with no built-in conv function."""
    if pad > 0:
        x = np.pad(x, pad)                       # zero padding
    N, F = x.shape[0], k.shape[0]
    n_out = (N - F) // stride + 1                # output size formula
    out = np.zeros((n_out, n_out), dtype=int)
    for i in range(n_out):                       # slide filter vertically
        for j in range(n_out):                   # slide filter horizontally
            r, c = i * stride, j * stride
            patch = x[r:r+F, c:c+F]              # current window
            out[i, j] = np.sum(patch * k)        # dot product at this location
    return out

out = conv2d(inp, filt, stride=1, pad=0)
print("Output feature map:\n", out)
print("Output shape:", out.shape)

out2 = conv2d(inp, filt, stride=2, pad=0)
print("\nStride = 2 output:\n", out2)
print("Output shape:", out2.shape)
