import pickle
import os

levels = 'batch'
fusion = 'moe'
bs = '2'
modality = 'TS_Text'
gating = 'laplace'
train_epochs = 8
num_experts = 12
hidden_size = 512
length = 'all'

filepath = f'/cis/home/xhan56/code/Multimodal-Transformer/run/TS_Text/ihm-{length}/{modality}/TS_48/Atten/Text_48/bioLongformer/1024/{fusion}/{levels}/{gating}/irregular_TS_64/irregular_Text_64/2e-05_2_3_0.0004_{train_epochs}_8_128_1_{bs}_{num_experts}_{hidden_size}'

with open(os.path.join(filepath, 'result.pkl'), 'rb') as f:
    result = pickle.load(f)

print(result)