
import numpy as np


### single out file ###
file_name = "out/computers.out"

file_content = open(file_name).readlines()

idxs = []
for i, line in enumerate(file_content):
    if not line.startswith("Epoch 1000:"): continue
    idxs.append(i)

metrics = []
for idx in idxs:
    for i in range(10):
        if file_content[idx+i].startswith('['):
            # print(file_content[idx+i])
            arrays = file_content[idx+i].strip()[1:-1]
            print(file_content[idx+i+1])
            break
    
    nums = [float(x.strip()) for x in arrays.split(",")]
    metrics.append(np.mean(nums))

# print(metrics)
for i in np.arange(0, len(metrics), 3):
    print(metrics[i:i+3])
