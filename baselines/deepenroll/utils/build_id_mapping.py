import pickle

diag_dataset = pickle.load(open('../../../data/diag_dataset', 'rb'))
prod_dataset = pickle.load(open('../../../data/prod_dataset', 'rb'))
proc_dataset = pickle.load(open('../../../data/proc_dataset', 'rb'))

code_to_id = {}
count = 0
for diag in diag_dataset:
  if diag == 'nan':
    code_to_id['nan_diag'] = count
    count += 1
  else:
    code_to_id[diag] = count
    count += 1

for prod in prod_dataset:
  if prod == 'nan':
    code_to_id['nan_prod'] = count
    count += 1
  else:
    code_to_id[prod] = count
    count += 1

for proc in proc_dataset:
  if proc == 'nan':
    code_to_id['nan_proc'] = count
    count += 1
  else:
    code_to_id[proc] = count
    count += 1

pickle.dump(code_to_id, open('../data/code_to_id', 'wb'))