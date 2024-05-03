import os

path = "release_model/E2FGVI-HQ-CVPR22"

with open(path + ".pth", "xb") as g:
    with open(path+".tmp1", "rb") as f:
        g.write(f.read(os.path.getsize(path+".tmp1")))
    with open(path+".tmp2", "rb") as f:
        g.write(f.read(os.path.getsize(path+".tmp2")))
        