import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the modules
from preprocessing.migraine_preprocessing import MigraineDataPipeline, EEGProcessor, WeatherConnector, SleepProcessor, StressProcessor
from core.pygmo_fusemoe import PyGMOFuseMoE, MigraineFuseMoE
from utils.config import MoEConfig

# ... existing code ... 