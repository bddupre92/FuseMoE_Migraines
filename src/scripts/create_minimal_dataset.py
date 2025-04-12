#!/usr/bin/env python3
# Script to create minimal processed datasets from MIMIC-IV demo data

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

print("Creating minimal MIMIC-IV dataset for FuseMOE...")

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(script_dir))
output_dir = os.path.join(base_dir, "Data/ihm")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Data will be saved to: {output_dir}")

# Create synthetic patient data
print("Creating synthetic patient data...")
num_patients = 200  # Use 200 patients
patient_ids = [f"p{10000000 + i}" for i in range(num_patients)]

# Create a dictionary to hold our synthetic data
synth_data = {
    'patient_id': [],
    'stay_id': [],
    'hadm_id': [],
    'icu_intime': [],
    'icu_outtime': [],
    'los_icu': [],
    'mortality_target': []
}

# Generate synthetic ICU stays
for i, patient_id in enumerate(patient_ids):
    # Multiple ICU stays for some patients
    num_stays = random.randint(1, 3)
    for j in range(num_stays):
        stay_id = f"stay_{i}_{j}"
        hadm_id = f"hadm_{i}_{j}"
        
        # Random admission time in the past year
        days_ago = random.randint(1, 365)
        icu_intime = datetime.now() - timedelta(days=days_ago)
        
        # Length of stay between 1 and 30 days
        los_days = random.randint(1, 30)
        icu_outtime = icu_intime + timedelta(days=los_days)
        
        # 15% mortality rate
        mortality = 1 if random.random() < 0.15 else 0
        
        # Add to our synthetic data dictionary
        synth_data['patient_id'].append(patient_id)
        synth_data['stay_id'].append(stay_id)
        synth_data['hadm_id'].append(hadm_id)
        synth_data['icu_intime'].append(icu_intime)
        synth_data['icu_outtime'].append(icu_outtime)
        synth_data['los_icu'].append(los_days)
        synth_data['mortality_target'].append(mortality)

# Create a DataFrame from the synthetic data
patients_df = pd.DataFrame(synth_data)

# Generate time series data for each patient
print("Generating synthetic time series data...")
timeseries_data = []

# Define feature names - FuseMOE expects 17 features
# We need to use exactly 17 features to match the model's expectation
vitals_features = ['HR', 'SBP', 'DBP', 'RESP', 'SpO2', 'Temp']
lab_features = [
    'Glucose', 'WBC', 'HGB', 'Sodium', 'Potassium', 'Chloride', 'BUN', 'Creatinine',
    'Feature9', 'Feature10', 'Feature11'  # Additional features to reach 17 total
]
all_features = vitals_features + lab_features
assert len(all_features) == 17, f"Expected 17 features, got {len(all_features)}"

# For each ICU stay, generate time series data
for _, stay in patients_df.iterrows():
    stay_id = stay['stay_id']
    los_days = stay['los_icu']
    
    # Calculate number of measurements based on length of stay
    num_measurements = max(int(los_days * 24 / 4), 6)  # Every 4 hours, minimum 6
    
    # Generate timestamps during the ICU stay
    timestamps = []
    start_time = stay['icu_intime']
    end_time = stay['icu_outtime']
    time_range = (end_time - start_time).total_seconds()
    
    # Generate synthetic data for this stay
    for i in range(num_measurements):
        # Random time during the ICU stay
        random_seconds = random.uniform(0, time_range)
        chart_time = start_time + timedelta(seconds=random_seconds)
        
        # Create a record for this timestamp
        record = {
            'stay_id': stay_id,
            'charttime': chart_time
        }
        
        # Generate values for each feature
        # Use normal distributions with realistic ranges for each feature
        for feature in all_features:
            if feature == 'HR':
                record[feature] = random.normalvariate(85, 15)
            elif feature == 'SBP':
                record[feature] = random.normalvariate(120, 20)
            elif feature == 'DBP':
                record[feature] = random.normalvariate(70, 10)
            elif feature == 'RESP':
                record[feature] = random.normalvariate(16, 4)
            elif feature == 'SpO2':
                record[feature] = min(100, random.normalvariate(96, 2))
            elif feature == 'Temp':
                record[feature] = random.normalvariate(37, 0.5)
            elif feature == 'Glucose':
                record[feature] = random.normalvariate(100, 25)
            elif feature == 'WBC':
                record[feature] = random.normalvariate(7.5, 2)
            elif feature == 'HGB':
                record[feature] = random.normalvariate(14, 2)
            elif feature == 'Sodium':
                record[feature] = random.normalvariate(140, 3)
            elif feature == 'Potassium':
                record[feature] = random.normalvariate(4, 0.5)
            elif feature == 'Chloride':
                record[feature] = random.normalvariate(100, 5)
            elif feature == 'BUN':
                record[feature] = random.normalvariate(15, 5)
            elif feature == 'Creatinine':
                record[feature] = random.normalvariate(1, 0.3)
            elif feature == 'Feature9':
                record[feature] = random.normalvariate(50, 10)
            elif feature == 'Feature10':
                record[feature] = random.normalvariate(25, 5)
            elif feature == 'Feature11':
                record[feature] = random.normalvariate(75, 15)
            
        timeseries_data.append(record)

# Create a DataFrame from the time series data
timeseries_df = pd.DataFrame(timeseries_data)
print(f"Generated {len(timeseries_df)} time series records for {len(patients_df)} patients")

# Generate clinical notes
print("Generating synthetic clinical notes...")
notes_data = []
note_types = ['Nursing', 'Physician', 'Discharge Summary', 'Radiology', 'ECG', 'Echo']

# Sample note templates
note_templates = {
    'Nursing': [
        "Patient is {condition}. Vital signs {vital_status}. {intervention}",
        "Nursing assessment: {assessment}. {plan}",
        "{time_of_day} nursing check: Patient {status}. {observation}"
    ],
    'Physician': [
        "History of present illness: {hpi}. Physical exam: {exam}. Assessment: {assessment}. Plan: {plan}",
        "Patient seen and examined. {findings}. Impression: {impression}. Recommended: {recommendation}",
        "Medical progress note: {progress}. Current management: {management}"
    ],
    'Discharge Summary': [
        "Discharge Diagnosis: {diagnosis}. Hospital Course: {course}. Discharge Medications: {medications}. Follow-up: {followup}",
        "Summary of Hospitalization: {summary}. Discharge Instructions: {instructions}",
        "Patient admitted on {admit_date} with {condition}. Treatment included {treatment}. Discharged home with {home_care}"
    ],
    'Radiology': [
        "Chest X-ray shows {findings}. Impression: {impression}",
        "CT scan of {area}: {findings}. No evidence of {negative_finding}. Impression: {impression}",
        "MRI of {area} demonstrates {findings}. Consistent with {diagnosis}"
    ],
    'ECG': [
        "12-lead ECG shows {rhythm} rhythm, rate {rate}. {finding}. Impression: {impression}",
        "ECG interpretation: {rhythm} at {rate} bpm. {intervals}. {assessment}",
        "Electrocardiogram: {finding}. No evidence of {negative_finding}. Conclusion: {conclusion}"
    ],
    'Echo': [
        "Echocardiogram: Left ventricular function {lv_function}. Ejection fraction {ef}%. {findings}",
        "Echo report: {chambers}. Valves: {valves}. No evidence of {negative_finding}",
        "Transthoracic echo: {finding}. LVEF {ef}%. {assessment}"
    ]
}

# Fill-in options for the templates
fill_ins = {
    'condition': ['stable', 'improving', 'deteriorating', 'critical but stable', 'comfortable'],
    'vital_status': ['within normal limits', 'slightly elevated', 'concerning', 'stable', 'improving'],
    'intervention': ['Continuing IV fluids.', 'Pain managed with medication.', 'Oxygen therapy continued.', 'Positioned for comfort.', 'Medication administered as ordered.'],
    'assessment': ['alert and oriented', 'responsive to treatment', 'showing signs of improvement', 'stable condition', 'requiring close monitoring'],
    'plan': ['Continue current treatment.', 'Adjust medication dosage.', 'Monitor for changes.', 'Prepare for discharge.', 'Consult specialist.'],
    'time_of_day': ['Morning', 'Afternoon', 'Evening', 'Night'],
    'status': ['resting comfortably', 'awake and alert', 'sleeping', 'in mild distress', 'stable'],
    'observation': ['No new complaints.', 'Pain level 3/10.', 'Appetite improving.', 'Ambulated in hallway with assistance.', 'Tolerated procedure well.'],
    'hpi': ['patient presented with acute onset of chest pain', 'patient with 3-day history of fever and cough', 'patient reports worsening shortness of breath', 'known diabetic with elevated blood glucose'],
    'exam': ['vital signs stable', 'lungs clear to auscultation', 'abdomen soft and non-tender', 'neurologically intact'],
    'impression': ['acute myocardial infarction', 'community-acquired pneumonia', 'diabetic ketoacidosis', 'normal study', 'no acute findings'],
    'recommendation': ['continue current management', 'adjust insulin dosage', 'add antibiotic therapy', 'obtain additional imaging', 'cardiology consultation'],
    'progress': ['patient showing clinical improvement', 'symptoms persist despite treatment', 'gradual resolution of presenting symptoms', 'stable condition without improvement'],
    'management': ['continuing IV antibiotics', 'insulin drip protocol', 'mechanical ventilation', 'vasopressor support', 'physical therapy'],
    'diagnosis': ['Acute MI', 'Pneumonia', 'CHF exacerbation', 'DKA', 'Sepsis'],
    'course': ['uneventful', 'complicated by acute kidney injury', 'required ICU management', 'responded well to treatment'],
    'medications': ['aspirin, lisinopril, metoprolol', 'insulin, metformin', 'ceftriaxone, azithromycin', 'furosemide, carvedilol'],
    'followup': ['primary care in 1 week', 'cardiology in 2 weeks', 'pulmonology as scheduled', 'return to ED for worsening symptoms'],
    'summary': ['patient admitted with respiratory distress requiring ventilatory support', 'patient treated for acute coronary syndrome', 'management of hyperglycemia and diabetic complications'],
    'instructions': ['take medications as prescribed', 'monitor blood glucose regularly', 'follow low sodium diet', 'continue physical therapy exercises'],
    'admit_date': ['yesterday', 'two days ago', 'last week', '3 days ago'],
    'treatment': ['IV antibiotics', 'coronary intervention', 'insulin therapy', 'mechanical ventilation'],
    'home_care': ['home oxygen', 'visiting nurse', 'physical therapy', 'wound care instructions'],
    'findings': ['patchy infiltrates in both lung bases', 'normal sinus rhythm', 'dilated cardiomyopathy', 'no acute abnormalities', 'mild degenerative changes'],
    'negative_finding': ['pneumothorax', 'fracture', 'mass', 'hemorrhage', 'infarction'],
    'area': ['chest', 'abdomen', 'brain', 'spine', 'pelvis'],
    'rhythm': ['normal sinus', 'atrial fibrillation', 'sinus tachycardia', 'sinus bradycardia', 'ventricular'],
    'rate': ['72', '88', '56', '110', '64'],
    'intervals': ['PR interval normal', 'prolonged QT interval', 'normal QRS duration', 'shortened PR interval'],
    'finding': ['ST elevation in anterior leads', 'T-wave inversions', 'bilateral pulmonary edema', 'hyperinflated lungs', 'mild cardiomegaly'],
    'conclusion': ['normal study', 'acute MI', 'COPD exacerbation', 'pneumonia', 'no significant change from prior'],
    'lv_function': ['normal', 'mildly reduced', 'moderately reduced', 'severely reduced'],
    'ef': ['60', '45', '35', '25', '55'],
    'chambers': ['normal chamber sizes', 'left atrial enlargement', 'right ventricular dilation', 'LV hypertrophy'],
    'valves': ['normal valve function', 'mitral regurgitation', 'aortic stenosis', 'tricuspid regurgitation']
}

# Generate synthetic notes for each patient
for _, stay in patients_df.iterrows():
    stay_id = stay['stay_id']
    patient_id = stay['patient_id']
    los_days = stay['los_icu']
    
    # Determine number of notes based on length of stay
    num_notes = max(int(los_days * 1.5), 3)  # At least 3 notes
    
    # Generate notes across the ICU stay
    start_time = stay['icu_intime']
    end_time = stay['icu_outtime']
    time_range = (end_time - start_time).total_seconds()
    
    for i in range(num_notes):
        # Random time during the ICU stay
        random_seconds = random.uniform(0, time_range)
        chart_time = start_time + timedelta(seconds=random_seconds)
        
        # Random note type
        note_type = random.choice(note_types)
        
        # Generate note text using templates
        template = random.choice(note_templates[note_type])
        
        # Replace placeholders with random choices
        note_text = template
        for placeholder, options in fill_ins.items():
            if '{' + placeholder + '}' in note_text:
                replacement = random.choice(options)
                note_text = note_text.replace('{' + placeholder + '}', replacement)
        
        # Add note to our collection
        notes_data.append({
            'stay_id': stay_id,
            'subject_id': patient_id,
            'charttime': chart_time,
            'category': note_type,
            'text': note_text
        })

notes_df = pd.DataFrame(notes_data)
print(f"Generated {len(notes_df)} clinical notes")

# Create the expected train/val/test splits
def create_dataset_splits(patients_df, timeseries_df, notes_df):
    # Shuffle patients and split into train/val/test (70/15/15)
    patient_ids = patients_df['patient_id'].unique()
    np.random.shuffle(patient_ids)
    
    train_size = int(len(patient_ids) * 0.7)
    val_size = int(len(patient_ids) * 0.15)
    
    train_ids = patient_ids[:train_size]
    val_ids = patient_ids[train_size:train_size+val_size]
    test_ids = patient_ids[train_size+val_size:]
    
    # Split patients dataframe
    train_patients = patients_df[patients_df['patient_id'].isin(train_ids)]
    val_patients = patients_df[patients_df['patient_id'].isin(val_ids)]
    test_patients = patients_df[patients_df['patient_id'].isin(test_ids)]
    
    # Get stay IDs for each split
    train_stays = train_patients['stay_id'].unique()
    val_stays = val_patients['stay_id'].unique()
    test_stays = test_patients['stay_id'].unique()
    
    # Split timeseries and notes based on stay IDs
    train_ts = timeseries_df[timeseries_df['stay_id'].isin(train_stays)]
    val_ts = timeseries_df[timeseries_df['stay_id'].isin(val_stays)]
    test_ts = timeseries_df[timeseries_df['stay_id'].isin(test_stays)]
    
    train_notes = notes_df[notes_df['stay_id'].isin(train_stays)]
    val_notes = notes_df[notes_df['stay_id'].isin(val_stays)]
    test_notes = notes_df[notes_df['stay_id'].isin(test_stays)]
    
    return {
        'train': {'patients': train_patients, 'timeseries': train_ts, 'notes': train_notes},
        'val': {'patients': val_patients, 'timeseries': val_ts, 'notes': val_notes},
        'test': {'patients': test_patients, 'timeseries': test_ts, 'notes': test_notes}
    }

# Create splits
splits = create_dataset_splits(patients_df, timeseries_df, notes_df)
print(f"Created data splits: {len(splits['train']['patients'])} train, {len(splits['val']['patients'])} val, {len(splits['test']['patients'])} test patients")

# Prepare data in the format expected by the FuseMOE model
def prepare_fusemoe_data(split_data):
    # Extract components
    patients = split_data['patients']
    timeseries = split_data['timeseries']
    notes = split_data['notes']
    
    # Structure required for FuseMOE
    fusemoe_data = []
    
    # For each patient stay
    for _, stay in patients.iterrows():
        stay_id = stay['stay_id']
        
        # Get all time series for this stay
        stay_ts = timeseries[timeseries['stay_id'] == stay_id]
        
        # Get all notes for this stay
        stay_notes = notes[notes['stay_id'] == stay_id]
        
        # Skip if no time series or notes available
        if len(stay_ts) == 0 or len(stay_notes) == 0:
            continue
        
        # Sort by charttime
        stay_ts = stay_ts.sort_values('charttime')
        stay_notes = stay_notes.sort_values('charttime')
        
        # Create a proper 2D array for time series data
        # Fixed size: (num_timesteps, num_features)
        # Using maximum of 48 timesteps to match the tt_max parameter
        num_timesteps = min(48, len(stay_ts))  # At most 48
        num_features = len(all_features)
        
        # Initialize arrays with zeros - ensure the size is exactly 48 rows
        ts_array = np.zeros((48, num_features))
        mask_array = np.zeros((48, num_features))
        
        # Fill with actual values where available (only for the rows that have data)
        for t_idx in range(min(num_timesteps, len(stay_ts))):
            row = stay_ts.iloc[t_idx]
            for f_idx, feature in enumerate(all_features):
                if feature in row and not pd.isna(row[feature]):
                    ts_array[t_idx, f_idx] = row[feature]
                    mask_array[t_idx, f_idx] = 1
        
        # Generate timestamps (ensure they're integers from 0 to 47)
        ts_tt = np.arange(48).tolist()
        
        # Regular time series - using shape (48, num_features*2)
        # FuseMOE expects exactly 48 x 34 (17 features x 2)
        reg_ts = np.zeros((48, num_features*2))
        
        # For each feature, fill in the regular time series with both the feature and its derivative
        for i in range(num_features):
            # Original feature values
            reg_ts[:, i] = ts_array[:, i]
            # Derivative/related feature (simplified as random values for the demo)
            reg_ts[:, i + num_features] = np.random.normal(0, 0.1, 48) + ts_array[:, i] * 0.1
        
        # Create IRREGULAR time series by randomly sampling points
        # We'll make the irregular time series sparser (30-70% of points missing randomly)
        irg_ts = np.copy(ts_array)
        irg_mask = np.copy(mask_array)
        
        # Create random mask - randomly drop 30-70% of points
        for t_idx in range(48):
            if np.random.random() < 0.5:  # 50% chance to modify this time point
                drop_percentage = np.random.uniform(0.3, 0.7)
                # Randomly select features to drop
                drop_indices = np.random.choice(
                    np.arange(num_features), 
                    size=int(num_features * drop_percentage), 
                    replace=False
                )
                # Set mask to 0 for dropped features
                irg_mask[t_idx, drop_indices] = 0
        
        # Take the last 5 notes (or pad with empty strings)
        last_notes = stay_notes.tail(5)
        note_texts = last_notes['text'].tolist()
        
        # Create text_time_to_end (ensure it's the same length as notes)
        text_time_to_end = np.linspace(0, 48, len(note_texts)).tolist() if len(note_texts) > 0 else []
        
        # Pad notes to have exactly 5
        while len(note_texts) < 5:
            note_texts.append("")
            if len(text_time_to_end) < 5:
                text_time_to_end.append(0)
        
        # Create the data item with essential fields only
        data_item = {
            'name': stay_id,
            'reg_ts': reg_ts,
            'irg_ts': irg_ts,  # Use our explicitly irregular time series
            'irg_ts_mask': irg_mask,  # Use our explicitly irregular mask
            'ts_tt': ts_tt,
            'text_data': note_texts,
            'text_time_to_end': text_time_to_end,
            'label': int(stay['mortality_target'])
        }
        
        fusemoe_data.append(data_item)
    
    return fusemoe_data

# Prepare data for each split
print("Preparing data in FuseMOE format...")
trainp2x_data = prepare_fusemoe_data(splits['train'])
valp2x_data = prepare_fusemoe_data(splits['val'])
testp2x_data = prepare_fusemoe_data(splits['test'])

# Save the processed data
print("Saving processed data...")
with open(os.path.join(output_dir, 'trainp2x_data.pkl'), 'wb') as f:
    pickle.dump(trainp2x_data, f)

with open(os.path.join(output_dir, 'valp2x_data.pkl'), 'wb') as f:
    pickle.dump(valp2x_data, f)

with open(os.path.join(output_dir, 'testp2x_data.pkl'), 'wb') as f:
    pickle.dump(testp2x_data, f)

# Also save the time series dataframe as expected by some preprocessing scripts
with open(os.path.join(output_dir, 'ts_labs_vitals_icu.pkl'), 'wb') as f:
    pickle.dump(timeseries_df, f)
    
print(f"Successfully saved processed data to {output_dir}")
print(f"Training data: {len(trainp2x_data)} examples")
print(f"Validation data: {len(valp2x_data)} examples")
print(f"Test data: {len(testp2x_data)} examples")

if __name__ == "__main__":
    print("Data generation complete. Ready for model training.") 