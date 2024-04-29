import pickle
import os

filepath = "../run/TS_CXR/ihm-48-cxr-notes-ecg/TS_CXR/TS_48/Atten/moe/['laplace']/joint/top_[4]/batch/irregular_TS_64/irregular_Text_64/0.0004_8_8_128_1_1_[12]_512/"
with open(os.path.join(filepath, 'result.pkl'), 'rb') as f:
    result = pickle.load(f)

print(result)