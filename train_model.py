import model
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm

word_dim = 768
mem_dim = 128
mlp_dim = 256
demo_dim = 3
batch_size = 384
epoch = 20
lr = 1e-4

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

embedding_dict = pickle.load(open('./data/embedding_dict','rb'))
prod_dataset = pickle.load(open('./data/prod_dataset','rb'))
proc_dataset = pickle.load(open('./data/proc_dataset','rb'))
diag_dataset = pickle.load(open('./data/diag_dataset','rb'))
prod_keys = pickle.load(open('./data/prod_keys','rb'))
proc_keys = pickle.load(open('./data/proc_keys','rb'))
diag_keys = pickle.load(open('./data/diag_keys','rb'))

train_demo_dataset = pickle.load(open('./data/train_demo_dataset','rb'))
train_ehr_dataset = pickle.load(open('./data/train_ehr_dataset','rb'))
train_id_dataset = pickle.load(open('./data/train_id_dataset','rb'))
train_trial_dataset = pickle.load(open('./data/train_trial_dataset','rb'))
train_label_dataset = pickle.load(open('./data/train_label_dataset','rb'))

valid_demo_dataset = pickle.load(open('./data/valid_demo_dataset','rb'))
valid_ehr_dataset = pickle.load(open('./data/valid_ehr_dataset','rb'))
valid_id_dataset = pickle.load(open('./data/valid_id_dataset','rb'))
valid_trial_dataset = pickle.load(open('./data/valid_trial_dataset','rb'))
valid_label_dataset = pickle.load(open('./data/valid_label_dataset','rb'))

def get_batch(loc, batch_size, mode):
  if mode == 'train':
    batch_id = train_id_dataset[loc:loc+batch_size]
    ehr = np.array(train_ehr_dataset, dtype=object)[batch_id]
    demo = np.array(train_demo_dataset)[batch_id]
    cur_trial = train_trial_dataset[loc:loc+batch_size]
    batch_label = train_label_dataset[loc:loc+batch_size]
  elif mode == 'valid':
    batch_id = valid_id_dataset[loc:loc+batch_size]
    ehr = np.array(valid_ehr_dataset, dtype=object)[batch_id]
    demo = np.array(valid_demo_dataset)[batch_id]
    cur_trial = valid_trial_dataset[loc:loc+batch_size]
    batch_label = valid_label_dataset[loc:loc+batch_size]
  else:
    batch_id = test_id_dataset[loc:loc+batch_size]
    ehr = np.array(test_ehr_dataset)[batch_id]
    demo = np.array(test_demo_dataset)[batch_id]
    cur_trial = test_trial_dataset[loc:loc+batch_size]
    batch_label = test_label_dataset[loc:loc+batch_size]
  
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

  return batch_ehr, batch_ehr_mask, batch_key, batch_demo, batch_criteria, batch_criteria_mask, batch_label

criteria_network = model.CriteriaEmbedding(word_dim, mlp_dim, mem_dim).to(device)
memory_network = model.EHRMemoryNetwork(word_dim, mem_dim, mlp_dim, demo_dim).to(device)
query_network = model.QueryNetwork(mem_dim, mlp_dim).to(device)
optimizer = torch.optim.Adam(list(criteria_network.parameters())+list(memory_network.parameters())+list(query_network.parameters()), lr=lr)

# Train
global_loss = 1e10
loss_list = []
for each_epoch in tqdm(range(epoch)):
  beam = int(25 - 21 * (each_epoch/(epoch-1)))
  for iteration in range(0, len(train_label_dataset), batch_size):
    criteria_network.train()
    memory_network.train()
    query_network.train()
    
    batch_ehr, batch_ehr_mask, batch_key, batch_demo, batch_criteria, batch_criteria_mask, batch_label = get_batch(iteration, batch_size, 'train')

    batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
    batch_ehr_mask = torch.tensor(batch_ehr_mask, dtype=torch.float32).to(device)
    batch_demo = torch.tensor(batch_demo, dtype=torch.float32).to(device)
    batch_criteria = torch.tensor(batch_criteria, dtype=torch.float32).to(device)
    batch_criteria_mask = torch.tensor(batch_criteria_mask, dtype=torch.float32).to(device)
    batch_label = torch.tensor(batch_label, dtype=torch.long).to(device)
  
    optimizer.zero_grad()

    loss, pred, att, response, query, memory = model.get_loss(batch_criteria, batch_criteria_mask, batch_ehr, batch_ehr_mask, batch_demo, batch_label, batch_key, query_network, memory_network, criteria_network, beam, False)
    
    loss.backward()
    optimizer.step()
    
    if iteration % (200*batch_size) == 0:
      print("Epoch %d, Iter %d: Loss:%.4f" % (each_epoch, iteration, loss))
    if iteration % (800*batch_size) == 0:
      if iteration == 0:
        continue
        
      criteria_network.eval()
      memory_network.eval()
      query_network.eval()
      with torch.no_grad():
        val_l = []
        for val_iter in range(0, len(valid_label_dataset), batch_size):
          batch_ehr, batch_ehr_mask, batch_key, batch_demo, batch_criteria, batch_criteria_mask, batch_label = get_batch(val_iter, batch_size, 'valid')

          batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
          batch_ehr_mask = torch.tensor(batch_ehr_mask, dtype=torch.float32).to(device)
          batch_demo = torch.tensor(batch_demo, dtype=torch.float32).to(device)
          batch_criteria = torch.tensor(batch_criteria, dtype=torch.float32).to(device)
          batch_criteria_mask = torch.tensor(batch_criteria_mask, dtype=torch.float32).to(device)
          batch_label = torch.tensor(batch_label, dtype=torch.long).to(device)
        
          optimizer.zero_grad()

          val_loss, val_pred, val_att, val_response, val_query, val_memory = model.get_loss(batch_criteria, batch_criteria_mask, batch_ehr, batch_ehr_mask, batch_demo, batch_label, batch_key, query_network, memory_network, criteria_network, beam, False)
    
          val_l.append((val_loss).detach().cpu().numpy())

        cur_val_loss = np.mean(val_l)
        print("Epoch %d validation: Loss:%.4f" % (each_epoch, cur_val_loss))
        if cur_val_loss < global_loss:
          global_loss = cur_val_loss
          state = {
              'embedding': criteria_network.state_dict(),
              'memory': memory_network.state_dict(),
              'query': query_network.state_dict(),
              'optimizer': optimizer.state_dict(),
              'iteration': iteration
          }
          torch.save(state, './save/model')
          print('\n------------ Save best model ------------\n')