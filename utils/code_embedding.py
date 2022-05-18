import os
import torch
import random
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def convert_examples_to_features(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    return input_ids,input_mask

local_rank = -1
fp16 = False
if local_rank == -1:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = 1
else:
  torch.cuda.set_device(local_rank)
  device = torch.device("cuda", local_rank)
  n_gpu = torch.cuda.device_count()
  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
  torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(local_rank != -1), fp16))
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)

model_loc = './pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/'
tokenizer = BertTokenizer.from_pretrained(model_loc, do_lower_case=True)
# Prepare model
cache_dir = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(local_rank))
model = BertModel.from_pretrained(model_loc,
          cache_dir=cache_dir)
if fp16:
    model.half()
model.to(device)
if local_rank != -1:
    try:
        from apex.parallel import DistributedDataParallel as DDP
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    model = DDP(model)
elif n_gpu > 1:
    model = torch.nn.DataParallel(model)

diag_df = pd.read_csv('../data/diagnosis_list.csv')
procedure_df = pd.read_csv('../data/procedure_list.csv')
product_df = pd.read_csv('../data/product_list.csv')

model.eval()
diag_dict = {}
diag_keys = {'nan': ['nan', 'nan', 'nan', 'nan']}
for idx, row in tqdm(diag_df.iterrows()):
  if str(row['diag_cd']) == 'nan':
    continue
  else:
    diag_dict[row['diag_cd']] = {}
    diag_keys[row['diag_cd']] = [row['diag_lvl1_desc'], row['diag_lvl2_desc'], row['diag_lvl4_desc'], row['diag_desc']]
    cur_id, cur_mask = convert_examples_to_features(row['diag_lvl1_desc'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    diag_dict[row['diag_cd']]['lv1'] = pooled_opt.squeeze(0).detach().cpu().numpy()

    cur_id, cur_mask = convert_examples_to_features(row['diag_lvl2_desc'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    diag_dict[row['diag_cd']]['lv2'] = pooled_opt.squeeze(0).detach().cpu().numpy()

    cur_id, cur_mask = convert_examples_to_features(row['diag_lvl4_desc'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    diag_dict[row['diag_cd']]['lv3'] = pooled_opt.squeeze(0).detach().cpu().numpy()

    cur_id, cur_mask = convert_examples_to_features(row['diag_desc'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    diag_dict[row['diag_cd']]['lv4'] = pooled_opt.squeeze(0).detach().cpu().numpy()
print('Finish diagnosis code')
pickle.dump(diag_dict, open('../data/diag_dict','wb'))
pickle.dump(diag_keys, open('../data/diag_keys','wb'))

proc_dict = {}
proc_keys = {'nan': ['nan', 'nan', 'nan', 'nan']}
for idx, row in tqdm(procedure_df.iterrows()):
  if str(row['prc_cd']) == 'nan':
    continue
  else:
    proc_dict[row['prc_cd']] = {}
    proc_keys[row['prc_cd']] = [row['prc_lvl1_desc'], row['prc_lvl2_desc'], row['prc_lvl4_desc'], row['prc_desc']]
    cur_id, cur_mask = convert_examples_to_features(row['prc_lvl1_desc'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    proc_dict[row['prc_cd']]['lv1'] = pooled_opt.squeeze(0).detach().cpu().numpy()

    cur_id, cur_mask = convert_examples_to_features(row['prc_lvl2_desc'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    proc_dict[row['prc_cd']]['lv2'] = pooled_opt.squeeze(0).detach().cpu().numpy()

    cur_id, cur_mask = convert_examples_to_features(row['prc_lvl4_desc'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    proc_dict[row['prc_cd']]['lv3'] = pooled_opt.squeeze(0).detach().cpu().numpy()

    cur_id, cur_mask = convert_examples_to_features(row['prc_desc'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    proc_dict[row['prc_cd']]['lv4'] = pooled_opt.squeeze(0).detach().cpu().numpy()
print('Finish diagnosis code')
pickle.dump(proc_dict,open('../data/proc_dict','wb'))
pickle.dump(proc_keys, open('../data/proc_keys','wb'))

prod_dict = {}
prod_keys = {'nan': ['nan', 'nan', 'nan', 'nan']}
for idx, row in tqdm(product_df.iterrows()):
  if str(row['product_id']) == 'nan':
    continue
  else:
    prod_dict[row['product_id']] = {}
    prod_keys[row['product_id']] = [row['usc_lvl2_desc'], row['usc_lvl3_desc'], row['usc_lvl4_desc'], row['mkted_prod_formltn_nm']]
    cur_id, cur_mask = convert_examples_to_features(row['usc_lvl2_desc'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    prod_dict[row['product_id']]['lv1'] = pooled_opt.squeeze(0).detach().cpu().numpy()

    cur_id, cur_mask = convert_examples_to_features(row['usc_lvl3_desc'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    prod_dict[row['product_id']]['lv2'] = pooled_opt.squeeze(0).detach().cpu().numpy()

    cur_id, cur_mask = convert_examples_to_features(row['usc_lvl4_desc'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    prod_dict[row['product_id']]['lv3'] = pooled_opt.squeeze(0).detach().cpu().numpy()

    cur_id, cur_mask = convert_examples_to_features(row['mkted_prod_formltn_nm'], tokenizer)
    cur_id = torch.tensor([cur_id], dtype=torch.long).to(device)
    cur_mask = torch.tensor([cur_mask], dtype=torch.long).to(device)
    encoder_opt, pooled_opt = model(cur_id, None, cur_mask, output_all_encoded_layers=False)
    prod_dict[row['product_id']]['lv4'] = pooled_opt.squeeze(0).detach().cpu().numpy()
print('Finish product code')
pickle.dump(prod_dict,open('../data/prod_dict','wb'))
pickle.dump(prod_keys,open('../data/prod_keys','wb'))