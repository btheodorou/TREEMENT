import os
import torch
import random
import pickle
import logging
import numpy as np
from tqdm import tqdm

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

trial_list = pickle.load(open('../data/clean_list','rb'))

criteria_dict = {}
for i in tqdm(range(len(trial_list))):
  cur_dict = {}
  cur_list = trial_list[i]['inclusion_list']
  cur_inc = []
  for j in range(len(cur_list)):
    cur_inc.append(cur_list[j])
  cur_dict['inclusion'] = cur_inc
  
  cur_list = trial_list[i]['exclusion_list'] 
  cur_exc = [] 
  for j in range(len(cur_list)):
    cur_exc.append(cur_list[j])
  cur_dict['exclusion'] = cur_exc

  criteria_dict[trial_list[i]['number']] = cur_dict
pickle.dump(criteria_dict,open('../data/criteria_dict','wb'))