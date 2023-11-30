import pickle
import os

levels = 'batch'
filepath = f'/cis/home/charr165/Documents/multimodal/results/ihm/TS/TS_48/Atten/layer3/{levels}/irregular_TS_64/irregular_Text_64/0.0004_6_8_128_1_8'
filepath = f'/cis/home/charr165/Documents/multimodal/results/ihm/TS/TS_48/Atten/layer1/batch_seq_feature/irregular_TS_64/irregular_Text_64/0.0004_8_8_128_1_8/'
filepath = '/cis/home/charr165/Documents/multimodal/results/ihm-48/TS/TS_48/Atten/layer3/batch_seq_feature/laplace/irregular_TS_64/irregular_Text_64/0.0004_8_8_128_1_8_12_512'

with open(os.path.join(filepath, 'result.pkl'), 'rb') as f:
    result = pickle.load(f)

print(result)