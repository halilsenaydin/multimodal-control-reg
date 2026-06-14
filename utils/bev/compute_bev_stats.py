import glob, numpy as np

s = np.zeros(3); ss = np.zeros(3); n = 0

for f in glob.glob("dataset_bev/train/*.npy"):
    a = np.load(f, mmap_mode="r")            # (N, 3, 224, 224)
    x = np.asarray(a, dtype=np.float64)
    s  += x.sum(axis=(0, 2, 3))
    ss += (x ** 2).sum(axis=(0, 2, 3))
    n  += x.shape[0] * x.shape[2] * x.shape[3]
    
mean = s / n
std  = np.sqrt(ss / n - mean ** 2)

print("BEV_MEAN =", mean.tolist())
print("BEV_STD  =", std.tolist())