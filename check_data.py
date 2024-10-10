data = []
import numpy as np
# def read_txt(self):
    # data = {}
with open('train_E3.txt', 'r', encoding='utf8') as input:
    for line in input:
        uid, prosody_path, timbre_path, content_path, target_path = line.strip().split('|')
        
        data.append([uid, prosody_path, timbre_path, content_path, target_path])

prosody_min = 1000
prosody_max = 0
prosody_mean = []
prosody_var = []
i = 0
prosodys = []
for uid, prosody_path, timbre_path, content_path, target_path in data[20000:]:
    prosody = np.load(target_path)
    # prosody_min = min(np.min(prosody), prosody_min)
    # prosody_max = max(np.max(prosody), prosody_max)
    print(prosody.shape)

    prosody_mean.append(np.mean(prosody, axis=-1))
    prosody_var.append(np.var(prosody, axis=-1))
    if i % 1000 == 0:
        print(i)
    i += 1

print(np.mean(prosody_mean), np.mean(prosody_var))
# print(np.mean(prosody_min), np.mean(prosody_max), np.mean(prosody_mean))
