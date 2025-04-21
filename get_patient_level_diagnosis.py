
# %%
import pandas as pd

# %%
metadata_filepath = "/lustre/fs1/home/cap5516.student2/LIDC-IDRI/metadata.csv"
diagnosis_filepath = "/lustre/fs1/home/cap5516.student2/tcia-diagnosis-data.csv"

# %%
metadata = pd.read_csv(metadata_filepath)
diagnosis = pd.read_csv(diagnosis_filepath)
# %%
# filter only by patients with diagnosis
filtered_metadata = metadata[metadata['Subject ID'].isin(diagnosis['TCIA Patient ID'])]

# %%
ct_filtered_metadata = filtered_metadata[filtered_metadata['SOP Class Name'] == 'CT Image Storage']
# %%
merged_df = pd.merge(
    ct_filtered_metadata,
    diagnosis,
    left_on='Subject ID',
    right_on='TCIA Patient ID',
    how='inner'
)

# %%
jsonl_data = []
for _, row in merged_df.iterrows():
    path = row['File Location']
    diagnosis_value = row['Diagnosis at the Patient Level\n0=Unknown\n1=benign or non-malignant disease\n2= malignant, primary lung cancer\n3 = malignant metastatic\n']
    
    entry = {
        "path": path,
        "diagnosis": int(diagnosis_value)
    }
    
    jsonl_data.append(entry)
# %%
with open("paths_with_diagnosis.txt", 'w') as f:
    for entry in jsonl_data:
        line = f"{entry['path']} {entry['diagnosis']}"
        f.write(line + '\n')
        print(line)
# %%


# %%
lines = []
with open('paths_with_diagnosis.txt', 'r') as file:
    for line in file:
        lines.append(line)
    
# %%

new_lines = []
for line in lines:
    diagnosis = line.strip()[-1]
    patient_id = line.strip().split('/')[2]
    
    new_line = patient_id + ' ' + diagnosis
    new_lines.append(new_line)
    
# %%
with open(f'patient_ids_with_diagnosis.txt', 'w') as file:
    for entry in new_lines:
        file.write(entry + '\n')
        print(line)

# %%
diagnosis_information = {}
with open(f'patient_ids_with_diagnosis.txt', 'r') as file:
    for entry in file:
        case_id = entry.strip()[:-2]
        diagnosis = entry.strip()[-1]
        diagnosis_information[case_id] = diagnosis
# %%
