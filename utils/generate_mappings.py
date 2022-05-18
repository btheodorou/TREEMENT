import pickle

trials = pickle.load(open('../data/trial_list', 'rb'))

phase_mapping = {}
disease_mapping = {}
for t in trials:
  if t['phase'] == 'Phase 1':
    phase_mapping[t['number']] = 1
  elif t['phase'] == 'Phase 2':
    phase_mapping[t['number']] = 2
  elif t['phase'] == 'Phase 3':
    phase_mapping[t['number']] = 3
  else:
    phase_mapping[t['number']] = -1

  condition = t['condition']
  category = []
  if 'Chronic' in condition:
    category.append('chronic')
  elif 'Cancer' in condition or 'Tumor' in condition or ('oma' in condition and 'Symptomatic Aortic Stenosis' not in condition and 'Symptomatic Vitreomacular Adhesion' not in condition):
    category.append('oncology')
  elif condition in ['Friedreich Ataxia', 'Advanced Gastric Cancer', 'IgA Nephropathy', 'Non-Squamous Non-Small Cell Lung Cancer', 'Hereditary Angioedema (HAE)', 'ALS Caused by Superoxide Dismutase 1 (SOD1) Mutation', 'Hemophilia A With Inhibitors', 'Neuromyelitis Optica', 'Hereditary Thrombotic Thrombocytopenic Purpura (TTP)', 'Huntington\'s Disease', 'Progressive Multifocal Leukoencephalopathy', 'Fabry Disease', 'Hilar Cholangiocarcinoma', 'Immune Reconstitution Inflammatory Syndrome', 'Spinal Muscular Atrophy', 'Aggressive Systemic Mastocytosis', 'Anxiety Neuroses', 'Dravet Syndrome', 'Refractory Generalized Myasthenia Gravis', 'Leber\'s Hereditary Optic Neuropathy (LHON)', 'Fabry Disease', 'Lennox-Gastaut Syndrome', 'Immune Reconstitution Inflammatory Syndrome']:
    category.append('rare')
  disease_mapping[t['number']] = category

pickle.dump(phase_mapping, open('../data/phase_mapping', 'wb'))
pickle.dump(disease_mapping, open('../data/disease_mapping', 'wb'))