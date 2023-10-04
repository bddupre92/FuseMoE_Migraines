import pickle
import os


filepath = '/home/xhan56/Multimodal-Transformer/run/TS_Text/ihm/TS_Text/TS_48/Atten/Text_48/bioLongformer/1024/cross_attn3/batch/irregular_TS_64/irregular_Text_64/2e-05_2_3_0.0004_6_8_128_1_6'

with open(os.path.join(filepath, 'result.pkl'), 'rb') as f:
    result = pickle.load(f)

print(result)