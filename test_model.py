from re import A
import yaml
import model
import torch
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.cm as cm
from sklearn import metrics
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from igraph import Graph, EdgeSeq, plot

def make_annotations(pos, text, font_size=8, font_color='rgb(50,50,50)'):
  annotations = []
  for k in range(len(pos)):
    annotations.append(
      dict(
        text=text[k], # or replace labels with a different list for the text within the circle
        x=pos[k][0], y=pos[k][1]-0.25-(0 if k%2==0 else 0.2),
        xref='x1', yref='y1',
        font=dict(color=font_color, size=font_size),
        showarrow=False)
      )
  return annotations

def plot_tree(g, treeList, cmap, graph_title, saveName):
  n_vertices = len(g.vs)
  layout = g.layout('rt')
  layout.rotate(180)
  labels = [t.key for t in treeList]
  attentions = [t.attention for t in treeList]
  edges = [e.tuple for e in g.es] # list of edges
  fig = go.Figure()
  for v in range(n_vertices):
    fig.add_trace(go.Scatter(x=[layout[v][0]], 
                             y=[layout[v][1]],
                             mode='markers', name=labels[v],
                             marker=dict(symbol='circle-dot',
                             size=50,
                             color='rgb'+str(cmap.to_rgba(attentions[v])[:-1]),    #'#DB4551',
                             line=dict(color='rgb(50,50,50)', width=1)),
                             text=[labels[v]],
                             hoverinfo='text',
                             opacity=1.0))

  for e in edges:
    fig.add_trace(go.Scatter(x=[layout[e[0]][0],layout[e[1]][0], None],
                             y=[layout[e[0]][1],layout[e[1]][1], None],
                             mode='lines',
                             line=dict(color='rgb(210,210,210)', width=1),
                             hoverinfo='none'))
  
  axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False)

  position = {k: layout[k] for k in range(n_vertices)}
  fig.update_layout(title=graph_title,
                annotations=make_annotations(position, labels),
                font_size=8,
                showlegend=False,
                xaxis=axis,
                yaxis=axis,
                margin=dict(l=40, r=40, b=85, t=100),
                hovermode='closest',
                plot_bgcolor='rgb(248,248,248)')
  # fig.show()
  fig.write_image(f"images/{saveName}.png")
 
word_dim = 768
mem_dim = 128
mlp_dim = 256
demo_dim = 3
batch_size = 128
lr = 1e-3

BEAM = 7
SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

local_rank = -1
fp16 = False
if local_rank == -1:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
else:
  torch.cuda.set_device(local_rank)
  device = torch.device("cuda", local_rank)
  n_gpu = 1
  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
  torch.distributed.init_process_group(backend='nccl')
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

criteria_dict = pickle.load(open('./data/criteria_dict','rb'))
embedding_dict = pickle.load(open('./data/embedding_dict','rb'))
prod_dataset = pickle.load(open('./data/prod_dataset','rb'))
proc_dataset = pickle.load(open('./data/proc_dataset','rb'))
diag_dataset = pickle.load(open('./data/diag_dataset','rb'))
prod_keys = pickle.load(open('./data/prod_keys','rb'))
proc_keys = pickle.load(open('./data/proc_keys','rb'))
diag_keys = pickle.load(open('./data/diag_keys','rb'))

test_demo_dataset = pickle.load(open('./data/test_demo_dataset','rb'))
test_ehr_dataset = pickle.load(open('./data/test_ehr_dataset','rb'))
test_id_dataset = pickle.load(open('./data/test_id_dataset','rb'))
test_trial_dataset = pickle.load(open('./data/test_trial_dataset','rb'))
test_label_dataset = pickle.load(open('./data/test_label_dataset','rb'))

phase_mapping = pickle.load(open('./data/phase_mapping', 'rb'))
disease_mapping = pickle.load(open('./data/disease_mapping', 'rb'))

def get_batch(loc, batch_size, mode):
  if mode == 'train':
    batch_id = train_id_dataset[loc:loc+batch_size]
    ehr = np.array(train_ehr_dataset)[batch_id]
    demo = np.array(train_demo_dataset)[batch_id]
    cur_trial = train_trial_dataset[loc:loc+batch_size]
    batch_label = train_label_dataset[loc:loc+batch_size]
  elif mode == 'valid':
    batch_id = valid_id_dataset[loc:loc+batch_size]
    ehr = np.array(valid_ehr_dataset)[batch_id]
    demo = np.array(valid_demo_dataset)[batch_id]
    cur_trial = valid_trial_dataset[loc:loc+batch_size]
    batch_label = valid_label_dataset[loc:loc+batch_size]
  else:
    batch_id = test_id_dataset[loc:loc+batch_size]
    ehr = np.array(test_ehr_dataset)[batch_id]
    demo = np.array(test_demo_dataset)[batch_id]
    cur_trial = test_trial_dataset[loc:loc+batch_size]
    batch_label = test_label_dataset[loc:loc+batch_size]
  
  batch_trial = [t[0] for t in cur_trial]
  batch_criteria_text = [criteria_dict[tmp_trial]['inclusion' if tmp_type == 'i' else 'exclusion'][tmp_ec] for (tmp_trial, tmp_type, tmp_ec) in cur_trial]

  batch_ehr = []
  batch_key = []
  batch_demo = []
  max_ts = 0
  for each_id in range(len(batch_id)):
    if len(ehr[each_id]) > max_ts:
      max_ts = len(ehr[each_id])
    
  batch_ehr_mask = []
  for each_id in range(len(batch_id)):    
    tmp_ehr = np.zeros((max_ts, 12, word_dim))
    tmp_mask = np.zeros(max_ts)
    tmp = [np.vstack((diag_dataset[ehr[each_id][0][0]],prod_dataset[ehr[each_id][0][1]],proc_dataset[ehr[each_id][0][2]]))]
    tmp_key = [(diag_keys[ehr[each_id][0][0]],prod_keys[ehr[each_id][0][1]],proc_keys[ehr[each_id][0][2]])]
    for j in range(1, len(ehr[each_id])):
      tmp.append(np.vstack((diag_dataset[ehr[each_id][j][0]],prod_dataset[ehr[each_id][j][1]],proc_dataset[ehr[each_id][j][2]])))
      tmp_key.append((diag_keys[ehr[each_id][j][0]],prod_keys[ehr[each_id][j][1]],proc_keys[ehr[each_id][j][2]]))
    tmp = np.array(tmp)
    tmp_ehr[:tmp.shape[0], :, :] = tmp
    tmp_mask[:tmp.shape[0]] = 1
    batch_ehr.append(tmp_ehr)
    batch_ehr_mask.append(tmp_mask)
    batch_key.append(tmp_key)
    batch_demo.append(demo[each_id])
    
  batch_criteria = []
  batch_criteria_mask = []
  max_seq = 0
  for each_id in range(len(batch_id)):
    tmp_trial, tmp_type, tmp_ec  = cur_trial[each_id]
    if tmp_type == 'i':
      if len(embedding_dict[tmp_trial]['inclusion'][tmp_ec]) > max_seq:
        max_seq = len(embedding_dict[tmp_trial]['inclusion'][tmp_ec])
    else:
      if len(embedding_dict[tmp_trial]['exclusion'][tmp_ec]) > max_seq:
        max_seq = len(embedding_dict[tmp_trial]['exclusion'][tmp_ec])
      
  for each_id in range(len(batch_id)):
    tmp_trial, tmp_type, tmp_ec  = cur_trial[each_id]
    tmp = np.zeros((max_seq, word_dim))
    tmp_mask = np.zeros((max_seq))
    if tmp_type == 'i':
      tmp[:len(embedding_dict[tmp_trial]['inclusion'][tmp_ec]), :] = embedding_dict[tmp_trial]['inclusion'][tmp_ec]
      tmp_mask[:len(embedding_dict[tmp_trial]['inclusion'][tmp_ec])] = 1 
    else:
      tmp[:len(embedding_dict[tmp_trial]['exclusion'][tmp_ec]), :] = embedding_dict[tmp_trial]['exclusion'][tmp_ec]
      tmp_mask[:len(embedding_dict[tmp_trial]['exclusion'][tmp_ec])] = 1
       
    batch_criteria.append(tmp)
    batch_criteria_mask.append(tmp_mask)

  batch_criteria = np.array(batch_criteria)
  batch_criteria_mask = np.array(batch_criteria_mask)
  batch_ehr = np.array(batch_ehr)
  batch_ehr_mask = np.array(batch_ehr_mask)
  batch_demo = np.array(batch_demo)
  batch_label = np.array(batch_label)

  return batch_ehr, batch_ehr_mask, batch_key, batch_demo, batch_criteria, batch_criteria_mask, batch_label, batch_id, batch_trial, batch_criteria_text

criteria_network = model.CriteriaEmbedding(word_dim, mlp_dim, mem_dim).to(device)
memory_network = model.EHRMemoryNetwork(word_dim, mem_dim, mlp_dim, demo_dim).to(device)
query_network = model.QueryNetwork(mem_dim, mlp_dim).to(device)
optimizer = torch.optim.Adam(list(criteria_network.parameters())+list(memory_network.parameters())+list(query_network.parameters()), lr=lr)

checkpoint = torch.load('./save/model', map_location=torch.device('cpu'))
criteria_network.load_state_dict(checkpoint['embedding'])
memory_network.load_state_dict(checkpoint['memory'])
query_network.load_state_dict(checkpoint['query'])
optimizer.load_state_dict(checkpoint['optimizer'])

loss_list = []
true_list = []
pred_list = []
response_list = []
query_list = []
id_list = []
trial_list = []
top_code_list = []
second_code_list = []
criteria_list = []
for iteration in tqdm(range(0, len(test_label_dataset), batch_size)):  
  criteria_network.eval()
  memory_network.eval()
  query_network.eval()
    
  batch_ehr, batch_ehr_mask, batch_key, batch_demo, batch_criteria, batch_criteria_mask, batch_label, batch_id, batch_trial, batch_criteria_text = get_batch(iteration, batch_size, 'test')

  batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
  batch_ehr_mask = torch.tensor(batch_ehr_mask, dtype=torch.float32).to(device)
  batch_demo = torch.tensor(batch_demo, dtype=torch.float32).to(device)
  batch_criteria = torch.tensor(batch_criteria, dtype=torch.float32).to(device)
  batch_criteria_mask = torch.tensor(batch_criteria_mask, dtype=torch.float32).to(device)
  batch_label = torch.tensor(batch_label, dtype=torch.long).to(device)
  
  optimizer.zero_grad()

  loss, pred, att, response, query, memory = model.get_loss(batch_criteria, batch_criteria_mask, batch_ehr, batch_ehr_mask, batch_demo, batch_label, batch_key, query_network, memory_network, criteria_network, BEAM, True)

  loss_list.append(loss.cpu().detach().numpy() if loss != 0 else 0)
  pred_list += list(pred.cpu().detach().numpy())
  true_list += list(batch_label.cpu().detach().numpy())
  response_list += list(response.cpu().detach().numpy())
  query_list += list(query.cpu().detach().numpy())
  id_list += batch_id
  trial_list += batch_trial
  criteria_list += batch_criteria_text

  for i in range(len(memory)):
    tree = memory[i]
    child_trees = []
    child_trees.extend(tree.children)
    for t in child_trees:
      child_trees.extend(t.children)
    
    child_trees.sort(key = lambda x: x.attention, reverse=True)
    top_code_list.append(child_trees[0].key)
    second_code_list.append(child_trees[1].key)

new_true_list = []
new_pred_list = []
new_round_list = []
new_id_list = []
new_trial_list = []
new_top_code_list = []
new_second_code_list = []
new_criteria_list = []
for i in range(len(pred_list)):
  if true_list[i] == 0:
    new_pred_list.append(1-pred_list[i][0])
    new_round_list.append(np.round(1-pred_list[i][0]))
    new_true_list.append(0)
    new_id_list.append(id_list[i])
    new_trial_list.append(trial_list[i])
    new_top_code_list.append(top_code_list[i])
    new_second_code_list.append(second_code_list[i])
    new_criteria_list.append(criteria_list[i])
  elif true_list[i] == 1:
    new_pred_list.append(pred_list[i][1])
    new_round_list.append(np.round(pred_list[i][1]))
    new_true_list.append(1)
    new_id_list.append(id_list[i])
    new_trial_list.append(trial_list[i])
    new_top_code_list.append(top_code_list[i])
    new_second_code_list.append(second_code_list[i])
    new_criteria_list.append(criteria_list[i])


intermediate = {'pred': new_pred_list, 'round': new_round_list, 'true': new_true_list, 'id': new_id_list, 'trial': new_trial_list}#, 'att': new_att_list, 'mem': new_mem_list, 'criteria': new_criteria_list}
pickle.dump(intermediate, open('save/intermediate.pkl', 'wb'))

intermediate = pickle.load(open('save/intermediate.pkl', 'rb'))
new_pred_list = intermediate['pred'] 
new_round_list = intermediate['round'] 
new_true_list = intermediate['true'] 
new_id_list = intermediate['id'] 
new_trial_list = intermediate['trial']

ndata = len(new_true_list)
n_boot = 1000



# Extract and display test metrics
print("CRITERIA-LEVEL RESULTS:")   
auroc = metrics.roc_auc_score(new_true_list, new_pred_list)
print('auroc: ', auroc)
acc = metrics.accuracy_score(new_true_list, new_round_list)
print('acc: ', acc)
(precisions, recalls, _) = metrics.precision_recall_curve(new_true_list, new_pred_list)
auprc = metrics.auc(recalls, precisions)
print('prc: ', auprc)
f1 = metrics.f1_score(new_true_list, new_round_list)
print('f1: ', f1)

auroc_bootstrap = []
for _ in tqdm(range(n_boot)):
  samples = random.choices(list(zip(new_true_list, new_pred_list)), k=ndata)
  auroc_bootstrap.append(metrics.roc_auc_score([i[0] for i in samples], [i[1] for i in samples]))
auroc_pm = (np.percentile(auroc_bootstrap, 97.5) - np.percentile(auroc_bootstrap, 2.5)) / 2
print('auroc_pm: ', auroc_pm)

acc_bootstrap = np.mean(np.random.choice([int(r == t) for (r,t) in zip(new_round_list, new_true_list)], size=(n_boot, ndata)), axis=1)
acc_pm = (np.percentile(acc_bootstrap, 97.5) - np.percentile(acc_bootstrap, 2.5)) / 2
print('acc_pm: ', acc_pm)

auprc_bootstrap = []
for _ in tqdm(range(n_boot)):
  samples = random.choices(list(zip(new_true_list, new_round_list)), k=ndata)
  (precisions, recalls, _) = metrics.precision_recall_curve([i[0] for i in samples], [i[1] for i in samples])
  auprc_bootstrap.append(metrics.auc(recalls, precisions))
auprc_pm = (np.percentile(auprc_bootstrap, 97.5) - np.percentile(auprc_bootstrap, 2.5)) / 2
print('auprc_pm: ', auprc_pm)

f1_bootstrap = []
for _ in tqdm(range(n_boot)):
  samples = random.choices(list(zip(new_true_list, new_round_list)), k=ndata)
  f1_bootstrap.append(metrics.f1_score([i[0] for i in samples], [i[1] for i in samples]))
f1_pm = (np.percentile(f1_bootstrap, 97.5) - np.percentile(f1_bootstrap, 2.5)) / 2
print('f1_pm: ', f1_pm)

results = {'auroc': auroc, 'acc': acc, 'auprc': auprc, 'f1': f1,
           'auroc_pm': auroc_pm, 'acc_pm': acc_pm, 'auprc_pm': auprc_pm, 'f1_pm': f1_pm}

pickle.dump(results, open('save/criteria_results.pkl', 'wb'))
print("\n\n")



trial_enrollments = {}
for i in range(len(new_true_list)):
  pair = (new_id_list[i], new_trial_list[i])
  if pair not in trial_enrollments:
    trial_enrollments[pair] = 1

  if new_true_list[i] != new_round_list[i]:
    trial_enrollments[pair] = 0

print("TRIAL-LEVEL RESULTS:")   
acc = np.mean(list(trial_enrollments.values()))
print('acc: ', acc)
acc_bootstrap = np.mean(np.random.choice(list(trial_enrollments.values()), size=(n_boot, ndata)), axis=1)
acc_pm = (np.percentile(acc_bootstrap, 97.5) - np.percentile(acc_bootstrap, 2.5)) / 2
print('acc_pm: ', acc_pm)
results = {'acc': acc, 'acc_pm': acc_pm}
pickle.dump(results, open('save/trial_results.pkl', 'wb'))
print("\n\n")



phase1 = []
phase2 = []
phase3 = []
for (_,t), v in trial_enrollments.items():
  if phase_mapping[t] == 1:
    phase1.append(v)
  elif phase_mapping[t] == 2:
    phase2.append(v)
  elif phase_mapping[t] == 3:
    phase3.append(v)

print("PHASE RESULTS:")   
acc1 = np.mean(phase1)
print('acc1: ', acc1)
acc_bootstrap1 = np.mean(np.random.choice(phase1, size=(n_boot, ndata)), axis=1)
acc_pm1 = (np.percentile(acc_bootstrap1, 97.5) - np.percentile(acc_bootstrap1, 2.5)) / 2
print('acc_pm1: ', acc_pm1)
acc2 = np.mean(phase2)
print('acc2: ', acc2)
acc_bootstrap2 = np.mean(np.random.choice(phase2, size=(n_boot, ndata)), axis=1)
acc_pm2 = (np.percentile(acc_bootstrap2, 97.5) - np.percentile(acc_bootstrap2, 2.5)) / 2
print('acc_pm2: ', acc_pm2)
acc3 = np.mean(phase3)
print('acc3: ', acc3)
acc_bootstrap3 = np.mean(np.random.choice(phase3, size=(n_boot, ndata)), axis=1)
acc_pm3 = (np.percentile(acc_bootstrap3, 97.5) - np.percentile(acc_bootstrap3, 2.5)) / 2
print('acc_pm3: ', acc_pm3)
results = {'acc1': acc1, 'acc_pm1': acc_pm1, 'acc2': acc2, 'acc_pm2': acc_pm2, 'acc3': acc3, 'acc_pm3': acc_pm3}
pickle.dump(results, open('save/phase_results.pkl', 'wb'))
print("\n\n")



chronic = []
oncology = []
rare = []
for (_,t), v in trial_enrollments.items():
  if 'chronic' in disease_mapping[t]:
    chronic.append(v)
  if 'oncology' in disease_mapping[t]:
    oncology.append(v)
  if 'rare' in disease_mapping[t]:
    rare.append(v)

print("DISEASE RESULTS:")   
acc_chronic = np.mean(chronic)
print('acc_chronic: ', acc_chronic)
acc_bootstrap_chronic = np.mean(np.random.choice(chronic, size=(n_boot, ndata)), axis=1)
acc_pm_chronic = (np.percentile(acc_bootstrap_chronic, 97.5) - np.percentile(acc_bootstrap_chronic, 2.5)) / 2
print('acc_pm_chronic: ', acc_pm_chronic)
acc_oncology = np.mean(oncology)
print('acc_oncology: ', acc_oncology)
acc_bootstrap_oncology = np.mean(np.random.choice(oncology, size=(n_boot, ndata)), axis=1)
acc_pm_oncology = (np.percentile(acc_bootstrap_oncology, 97.5) - np.percentile(acc_bootstrap_oncology, 2.5)) / 2
print('acc_pm_oncology: ', acc_pm_oncology)
acc_rare = np.mean(rare)
print('acc_rare: ', acc_rare)
acc_bootstrap_rare = np.mean(np.random.choice(rare, size=(n_boot, ndata)), axis=1)
acc_pm_rare = (np.percentile(acc_bootstrap_rare, 97.5) - np.percentile(acc_bootstrap_rare, 2.5)) / 2
print('acc_pm_rare: ', acc_pm_rare)
results = {'acc_chronic': acc_chronic, 'acc_pm_chronic': acc_pm_chronic, 'acc_oncology': acc_oncology, 'acc_c_oncology2': acc_pm_oncology, 'acc_rare': acc_rare, 'acc_pm_rare': acc_pm_rare}
pickle.dump(results, open('save/disease_results.pkl', 'wb'))



print("TYPE LENGTHS:")   
phase1_len = len(phase1)
print('phase1_len: ', phase1_len)
phase2_len = len(phase2)
print('phase2_len: ', phase2_len)
phase3_len = len(phase3)
print('phase3_len: ', phase3_len)
chronic_len = len(chronic)
print('chronic_len: ', chronic_len)
oncology_len = len(oncology)
print('oncology_len: ', oncology_len)
rare_len = len(rare)
print('rare_len: ', rare_len)
results = {'phase1_len': phase1_len, 'phase2_len': phase2_len, 'phase3_len': phase3_len, 'chronic_len': chronic_len, 'oncology_len': oncology_len, 'rare_len': rare_len,}
pickle.dump(results, open('save/type_results.pkl', 'wb'))
print("\n\n")



print("INTERPRETABILITY CHECK:") 
df = pd.DataFrame({"Criteria": new_criteria_list, "Label": new_true_list, "Code 1": new_top_code_list, "Code 2": new_second_code_list})
df.to_csv('save/interpretability.csv')
print("\n\n")



print("INTERPRETABILITY VISUALIZATION:") 
NUM_TREES = 10000
PRUNING_THRESHOLD = 0.0
# Extract and display some memory trees
true_list = []
pred_list = []
mem_list = []
criteria_list = []
for i in range(NUM_TREES):  
  criteria_network.eval()
  memory_network.eval()
  query_network.eval()
    
  idx = np.random.randint(len(test_label_dataset))

  batch_ehr, batch_ehr_mask, batch_key, batch_demo, batch_criteria, batch_criteria_mask, batch_label, batch_id, batch_trial, batch_criteria_text = get_batch(idx, 2, 'test')

  batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
  batch_ehr_mask = torch.tensor(batch_ehr_mask, dtype=torch.float32).to(device)
  batch_demo = torch.tensor(batch_demo, dtype=torch.float32).to(device)
  batch_criteria = torch.tensor(batch_criteria, dtype=torch.float32).to(device)
  batch_criteria_mask = torch.tensor(batch_criteria_mask, dtype=torch.float32).to(device)
  batch_label = torch.tensor(batch_label, dtype=torch.long).to(device)
  
  optimizer.zero_grad()

  loss, pred, att, response, query, memory = model.get_loss(batch_criteria, batch_criteria_mask, batch_ehr, batch_ehr_mask, batch_demo, batch_label, batch_key, query_network, memory_network, criteria_network, BEAM, True)

  pred_list += list(pred.cpu().detach().numpy())
  true_list += list(batch_label.cpu().detach().numpy())
  mem_list += memory
  criteria_list += batch_criteria_text

new_true_list = []
new_pred_list = []
new_round_list = []
new_mem_list = []
new_criteria_list = []
for i in range(len(pred_list)):
  if true_list[i] == 0:
    new_pred_list.append(1-pred_list[i][0])
    new_round_list.append(np.round(1-pred_list[i][0]))
    new_true_list.append(0)
    new_mem_list.append(mem_list[i])
    new_criteria_list.append(criteria_list[i])
  elif true_list[i] == 1:
    new_pred_list.append(pred_list[i][1])
    new_round_list.append(np.round(pred_list[i][1]))
    new_true_list.append(1)
    new_mem_list.append(mem_list[i])
    new_criteria_list.append(criteria_list[i])

for idx in range(len(new_mem_list)):
  tree = new_mem_list[idx]
  criteria = new_criteria_list[idx]
  label = new_true_list[idx]
  round = new_round_list[idx]
  att = new_att_list[idx]
  if label != round or label:
    continue

  title = f"{criteria} - {'Mismatch' if label else 'Match'}"

  # Standard Version
  treeList = [tree]
  edges = []
  for i, t in enumerate(treeList):
    for j, c in enumerate(t.children):
      edges.append((i,len(treeList)))
      treeList.append(c)
  g = Graph(edges)
  maxIdx = np.argmax([t.attention for t in treeList])
  maxTree = [t for t in treeList][maxIdx]
  maxAttention = maxTree.attention * 1.2
  maxLabel = maxTree.key
  if maxLabel == 'Demographics':
    continue

  norm = mpl.colors.Normalize(vmin=0, vmax=maxAttention)
  cmap = cm.gist_heat
  cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
  plot_tree(g, treeList, cmap, title, f"{idx}")

  # Pruned Version
  treeList = [tree]
  edges = []
  for i, t in enumerate(treeList):
    for j, c in enumerate(t.children):
      totalAttention = c.attention
      descendants = c.children
      for d in descendants:
        descendants.extend(d.children)
      totalAttention += np.sum([d.attention for d in descendants])
      if totalAttention > PRUNING_THRESHOLD:
        edges.append((i,len(treeList)))
        treeList.append(c)
  g = Graph(edges)
  plot_tree(g, treeList, cmap, title, f"{idx}-2")