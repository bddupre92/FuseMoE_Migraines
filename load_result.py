import pickle
import os

levels = 'batch_seq'
filepath = f'/cis/home/xhan56/code/Multimodal-Transformer/run/TS_Text/ihm/TS_Text/TS_48/Atten/Text_48/bioLongformer/1024/cross_attn3/{levels}/irregular_TS_64/irregular_Text_64/2e-05_2_3_0.0004_6_8_128_1_3'

with open(os.path.join(filepath, 'result.pkl'), 'rb') as f:
    result = pickle.load(f)

print(result)