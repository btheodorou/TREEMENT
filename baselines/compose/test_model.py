import torch
import numpy as np
import random
import logging
import pickle
import model
from sklearn import metrics
import random
from tqdm import tqdm

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

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
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(local_rank != -1), fp16))
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

embedding_dict = pickle.load(open('../../data/embedding_dict','rb'))
prod_dataset = pickle.load(open('../../data/prod_dataset','rb'))
proc_dataset = pickle.load(open('../../data/proc_dataset','rb'))
diag_dataset = pickle.load(open('../../data/diag_dataset','rb'))

test_demo_dataset = pickle.load(open('../../data/test_demo_dataset','rb'))
test_ehr_dataset = pickle.load(open('../../data/test_ehr_dataset','rb'))
test_id_dataset = pickle.load(open('../../data/test_id_dataset','rb'))
test_trial_dataset = pickle.load(open('../../data/test_trial_dataset','rb'))
test_label_dataset = pickle.load(open('../../data/test_label_dataset','rb'))

phase_mapping = pickle.load(open('../../data/phase_mapping', 'rb'))
disease_mapping = pickle.load(open('../../data/disease_mapping', 'rb'))

# def get_batch(loc, batch_size, mode):
#   if mode == 'train':
#     batch_id = train_id_dataset[loc:loc+batch_size]
#     ehr = np.array(train_ehr_dataset)[batch_id]
#     demo = np.array(train_demo_dataset)[batch_id]
#     cur_trial = train_trial_dataset[loc:loc+batch_size]
#     batch_label = train_label_dataset[loc:loc+batch_size]
#   elif mode == 'valid':
#     batch_id = valid_id_dataset[loc:loc+batch_size]
#     ehr = np.array(valid_ehr_dataset)[batch_id]
#     demo = np.array(valid_demo_dataset)[batch_id]
#     cur_trial = valid_trial_dataset[loc:loc+batch_size]
#     batch_label = valid_label_dataset[loc:loc+batch_size]
#   else:
#     batch_id = test_id_dataset[loc:loc+batch_size]
#     ehr = np.array(test_ehr_dataset)[batch_id]
#     demo = np.array(test_demo_dataset)[batch_id]
#     cur_trial = test_trial_dataset[loc:loc+batch_size]
#     batch_label = test_label_dataset[loc:loc+batch_size]
  
#   batch_trial = [t[0] for t in cur_trial]

#   batch_ehr = []
#   batch_demo = []
#   max_ts = 0
#   for each_id in range(len(batch_id)):
#     if len(ehr[each_id]) > max_ts:
#       max_ts = len(ehr[each_id])
    
#   batch_ehr_mask = []
#   for each_id in range(len(batch_id)):    
#     tmp_ehr = np.zeros((max_ts, 12, word_dim))
#     tmp_mask = np.zeros(max_ts)
#     tmp = [np.vstack((diag_dataset[ehr[each_id][0][0]],prod_dataset[ehr[each_id][0][1]],proc_dataset[ehr[each_id][0][2]]))]
#     for j in range(1, len(ehr[each_id])):
#       tmp.append(np.vstack((diag_dataset[ehr[each_id][j][0]],prod_dataset[ehr[each_id][j][1]],proc_dataset[ehr[each_id][j][2]])))
#     tmp = np.array(tmp)
#     tmp_ehr[:tmp.shape[0], :, :] = tmp
#     tmp_mask[:tmp.shape[0]] = 1
#     batch_ehr.append(tmp_ehr)
#     batch_ehr_mask.append(tmp_mask)
#     batch_demo.append(demo[each_id])
    
#   batch_criteria = []
#   batch_criteria_mask = []
#   max_seq = 0
#   for each_id in range(len(batch_id)):
#     tmp_trial, tmp_type, tmp_ec  = cur_trial[each_id]
#     if tmp_type == 'i':
#       if len(embedding_dict[tmp_trial]['inclusion'][tmp_ec]) > max_seq:
#         max_seq = len(embedding_dict[tmp_trial]['inclusion'][tmp_ec])
#     else:
#       if len(embedding_dict[tmp_trial]['exclusion'][tmp_ec]) > max_seq:
#         max_seq = len(embedding_dict[tmp_trial]['exclusion'][tmp_ec])
      
#   for each_id in range(len(batch_id)):
#     tmp_trial, tmp_type, tmp_ec  = cur_trial[each_id]
#     tmp = np.zeros((max_seq, word_dim))
#     tmp_mask = np.zeros((max_seq))
#     if tmp_type == 'i':
#       tmp[:len(embedding_dict[tmp_trial]['inclusion'][tmp_ec]), :] = embedding_dict[tmp_trial]['inclusion'][tmp_ec]
#       tmp_mask[:len(embedding_dict[tmp_trial]['inclusion'][tmp_ec])] = 1 
#     else:
#       tmp[:len(embedding_dict[tmp_trial]['exclusion'][tmp_ec]), :] = embedding_dict[tmp_trial]['exclusion'][tmp_ec]
#       tmp_mask[:len(embedding_dict[tmp_trial]['exclusion'][tmp_ec])] = 1
       
#     batch_criteria.append(tmp)
#     batch_criteria_mask.append(tmp_mask)

#   batch_criteria = np.array(batch_criteria)
#   batch_criteria_mask = np.array(batch_criteria_mask)
#   batch_ehr = np.array(batch_ehr)
#   batch_ehr_mask = np.array(batch_ehr_mask)
#   batch_demo = np.array(batch_demo)
#   batch_label = np.array(batch_label)

#   return batch_ehr, batch_ehr_mask, batch_demo, batch_criteria, batch_criteria_mask, batch_label, batch_id, batch_trial

# word_dim = 768
# conv_dim = 128
# mem_dim = 320
# mlp_dim = 512
# demo_dim = 3
# class_dim = 477
# margin = 1

# batch_size = 256
# epoch = 20
# lr=1e-3

# embedding_network = model.ECEmbedding(word_dim, conv_dim, mem_dim).to(device)
# memory_network = model.EHRMemoryNetwork(word_dim, mem_dim, demo_dim).to(device)
# query_network = model.QueryNetwork(mem_dim, conv_dim, mlp_dim).to(device)
# optimizer = torch.optim.Adam(list(embedding_network.parameters())+list(memory_network.parameters())+list(query_network.parameters()), lr=lr)

# checkpoint = torch.load('./save/model')
# embedding_network.load_state_dict(checkpoint['embedding'])
# memory_network.load_state_dict(checkpoint['memory'])
# query_network.load_state_dict(checkpoint['query'])
# optimizer.load_state_dict(checkpoint['optimizer'])

# loss_list = []
# pred_list = []
# res_list =[]
# query_list = []
# true_list = []
# sim_list = []
# att_list = []
# id_list = []
# trial_list = []
# for iteration in range(0, len(test_trial_dataset), batch_size):
#   embedding_network.eval()
#   memory_network.eval()
#   query_network.eval()
    
#   batch_ehr, batch_ehr_mask, batch_demo, batch_criteria, batch_criteria_mask, batch_label, batch_id, batch_trial = get_batch(iteration, batch_size, 'test')
  
#   batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
#   batch_ehr_mask = torch.tensor(batch_ehr_mask, dtype=torch.float32).to(device)
#   batch_demo = torch.tensor(batch_demo, dtype=torch.float32).to(device)
#   batch_criteria = torch.tensor(batch_criteria, dtype=torch.float32).to(device)
#   batch_criteria_mask = torch.tensor(batch_criteria_mask, dtype=torch.float32).to(device)
#   batch_label = torch.tensor(batch_label, dtype=torch.long).to(device)
  
#   optimizer.zero_grad()

#   loss, sim, pred, att, res, query = model.get_loss(batch_criteria, batch_criteria_mask, 
#             batch_ehr, batch_ehr_mask, batch_demo, batch_label,
#             query_network, memory_network, embedding_network, device)
#   att_list += list(att.cpu().detach().numpy())
#   loss_list.append(loss.cpu().detach().numpy() if loss != 0 else 0)
#   pred_list += list(pred.cpu().detach().numpy())
#   true_list += list(batch_label.cpu().detach().numpy())
#   res_list += list(res.cpu().detach().numpy())
#   query_list += list(query.cpu().detach().numpy())
#   id_list += batch_id
#   trial_list += batch_trial
#   sim_list.append(sim.cpu().detach().numpy())
#   if iteration % 500 == 0:
#     print('%d / %d'%(iteration, len(test_trial_dataset)))
  
# new_true = []
# new_pred = []
# new_round = []
# new_id = []
# new_trial = []
# for i in range(len(pred_list)):
#   if true_list[i] == 0:
#     new_pred.append(1-pred_list[i][0])
#     new_round.append(np.round(1-pred_list[i][0]))
#     new_true.append(0)
#     new_id.append(id_list[i])
#     new_trial.append(trial_list[i])
#   elif true_list[i] == 1:
#     new_pred.append(pred_list[i][1])
#     new_round.append(np.round(pred_list[i][1]))
#     new_true.append(1)
#     new_id.append(id_list[i])
#     new_trial.append(trial_list[i])

# intermediate = {'pred': new_pred, 'round': new_round, 'true': new_true, 'id': new_id, 'trial': new_trial}
# pickle.dump(intermediate, open('save/intermediate.pkl', 'wb'))

intermediate = pickle.load(open('save/intermediate.pkl', 'rb'))
new_pred = intermediate['pred']
new_round = intermediate['round']
new_true = intermediate['true']
new_id = intermediate['id']
new_trial = intermediate['trial']

ndata = len(new_true)
n_boot = 1000



# print("CRITERIA-LEVEL RESULTS:")   
# auroc = metrics.roc_auc_score(new_true, new_pred)
# print('auroc: ', auroc)
# acc = metrics.accuracy_score(new_true, new_round)
# print('acc: ', acc)
# (precisions, recalls, _) = metrics.precision_recall_curve(new_true, new_pred)
# auprc = metrics.auc(recalls, precisions)
# print('auprc: ', auprc)
# f1 = metrics.f1_score(new_true, new_round)
# print('f1: ', f1)

# auroc_bootstrap = []
# for _ in tqdm(range(n_boot)):
#   samples = random.choices(list(zip(new_true, new_pred)), k=ndata)
#   auroc_bootstrap.append(metrics.roc_auc_score([i[0] for i in samples], [i[1] for i in samples]))
# auroc_pm = (np.percentile(auroc_bootstrap, 97.5) - np.percentile(auroc_bootstrap, 2.5)) / 2
# print('auroc_pm: ', auroc_pm)

# acc_bootstrap = np.mean(np.random.choice([int(r == t) for (r,t) in zip(new_round, new_true)], size=(n_boot, ndata)), axis=1)
# acc_pm = (np.percentile(acc_bootstrap, 97.5) - np.percentile(acc_bootstrap, 2.5)) / 2
# print('acc_pm: ', acc_pm)

# auprc_bootstrap = []
# for _ in tqdm(range(n_boot)):
#   samples = random.choices(list(zip(new_true, new_round)), k=ndata)
#   (precisions, recalls, _) = metrics.precision_recall_curve([i[0] for i in samples], [i[1] for i in samples])
#   auprc_bootstrap.append(metrics.auc(recalls, precisions))
# auprc_pm = (np.percentile(auprc_bootstrap, 97.5) - np.percentile(auprc_bootstrap, 2.5)) / 2
# print('auprc_pm: ', auprc_pm)

# f1_bootstrap = []
# for _ in tqdm(range(n_boot)):
#   samples = random.choices(list(zip(new_true, new_round)), k=ndata)
#   f1_bootstrap.append(metrics.f1_score([i[0] for i in samples], [i[1] for i in samples]))
# f1_pm = (np.percentile(f1_bootstrap, 97.5) - np.percentile(f1_bootstrap, 2.5)) / 2
# print('f1_pm: ', f1_pm)

# results = {'auroc': auroc, 'acc': acc, 'auprc': auprc, 'f1': f1,
#            'auroc_pm': auroc_pm, 'acc_pm': acc_pm, 'auprc_pm': auprc_pm, 'f1_pm': f1_pm}

# pickle.dump(results, open('save/criteria_results.pkl', 'wb'))
# print("\n\n")



trial_enrollments = {}
for i in range(len(new_true)):
  pair = (new_id[i], new_trial[i])
  if pair not in trial_enrollments:
    trial_enrollments[pair] = 1
  if new_true[i] != new_round[i]:
    trial_enrollments[pair] = 0

# print("TRIAL-LEVEL RESULTS:")   
# acc = np.mean(list(trial_enrollments.values()))
# print('acc: ', acc)
# acc_bootstrap = np.mean(np.random.choice(list(trial_enrollments.values()), size=(n_boot, ndata)), axis=1)
# acc_pm = (np.percentile(acc_bootstrap, 97.5) - np.percentile(acc_bootstrap, 2.5)) / 2
# print('acc_pm: ', acc_pm)
# results = {'acc': acc, 'acc_pm': acc_pm}
# pickle.dump(results, open('save/trial_results.pkl', 'wb'))
# print("\n\n")



# phase1 = []
# phase2 = []
# phase3 = []
# for (_,t), v in trial_enrollments.items():
#   if phase_mapping[t] == 1:
#     phase1.append(v)
#   elif phase_mapping[t] == 2:
#     phase2.append(v)
#   elif phase_mapping[t] == 3:
#     phase3.append(v)

# print("PHASE RESULTS:")   
# acc1 = np.mean(phase1)
# print('acc1: ', acc1)
# acc_bootstrap1 = np.mean(np.random.choice(phase1, size=(n_boot, ndata)), axis=1)
# acc_pm1 = (np.percentile(acc_bootstrap1, 97.5) - np.percentile(acc_bootstrap1, 2.5)) / 2
# print('acc_pm1: ', acc_pm1)
# acc2 = np.mean(phase2)
# print('acc2: ', acc2)
# acc_bootstrap2 = np.mean(np.random.choice(phase2, size=(n_boot, ndata)), axis=1)
# acc_pm2 = (np.percentile(acc_bootstrap2, 97.5) - np.percentile(acc_bootstrap2, 2.5)) / 2
# print('acc_pm2: ', acc_pm2)
# acc3 = np.mean(phase3)
# print('acc3: ', acc3)
# acc_bootstrap3 = np.mean(np.random.choice(phase3, size=(n_boot, ndata)), axis=1)
# acc_pm3 = (np.percentile(acc_bootstrap3, 97.5) - np.percentile(acc_bootstrap3, 2.5)) / 2
# print('acc_pm3: ', acc_pm3)
# results = {'acc1': acc1, 'acc_pm1': acc_pm1, 'acc2': acc2, 'acc_pm2': acc_pm2, 'acc3': acc3, 'acc_pm3': acc_pm3}
# pickle.dump(results, open('save/phase_results.pkl', 'wb'))
# print("\n\n")



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