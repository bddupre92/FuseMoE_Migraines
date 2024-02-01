import pickle
import os

# levels = 'batch'
# fusion = 'moe'
# bs = '1'
# modality = 'TS_CXR'
# gating = 'softmax'
# train_epochs = 8
# num_experts = 12
# hidden_size = 512
# task = 'pheno-all-cxr-notes'

# filepath = f'/cis/home/xhan56/code/Multimodal-Transformer/run/{modality}/{task}/{modality}/TS_48/Atten/Text_48/bioLongformer/1024/{fusion}/{levels}/{gating}/irregular_TS_64/irregular_Text_64/2e-05_2_3_0.0004_{train_epochs}_8_128_1_{bs}_{num_experts}_{hidden_size}'
# fp_no_text = f'/cis/home/xhan56/code/Multimodal-Transformer/run/{modality}/{task}/{modality}/TS_48/Atten/{levels}/{gating}/irregular_TS_64/irregular_Text_64/0.0004_{train_epochs}_8_128_1_{bs}_{num_experts}_{hidden_size}'

filepath = "run/TS_CXR_Text_ECG/pheno-all-cxr-notes-ecg/TS_CXR_Text_ECG/TS_48/Atten/Text_48/bioLongformer/1024/moe/laplace/joint/top_4/batch/irregular_TS_64/irregular_Text_64/use_pt_text_embeddings/2e-05_2_3_0.0004_8_8_128_1_1_12_512/"
with open(os.path.join(filepath, 'result.pkl'), 'rb') as f:
    result = pickle.load(f)

print(result)