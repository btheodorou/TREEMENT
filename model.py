import torch
from torch import nn
from typing import List
   
class ResidualBlock(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(ResidualBlock, self).__init__()
    self.layer1 = nn.Linear(input_dim, hidden_dim)
    self.layer2 = nn.Linear(hidden_dim, input_dim)
    self.final_layer = nn.Linear(input_dim, output_dim)
    
  def forward(self, input):
    out = self.layer1(input)
    out = torch.relu(out)
    out = self.layer2(out)
    out += input
    out = self.final_layer(out)
    return out
  
  
  
class CriteriaEmbedding(nn.Module):
  def __init__(self, word_dim, hidden_dim, mem_dim):
    super(CriteriaEmbedding, self).__init__()
    #Input: batch_size * demo_dim
    # Network for word features
    self.word_encoder = nn.TransformerEncoderLayer(word_dim, nhead=4, batch_first=True)
    self.word_encoder2 = nn.TransformerEncoderLayer(word_dim, nhead=4, batch_first=True)
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.skip_layer = ResidualBlock(word_dim, hidden_dim, mem_dim)
    self.skip_layer2 = ResidualBlock(mem_dim, hidden_dim, mem_dim)
  def forward(self, input, mask):
    word_embds = self.word_encoder(input, src_key_padding_mask=(1.0 - mask))
    word_embds = self.word_encoder2(word_embds, src_key_padding_mask=(1.0 - mask))
    masked_word_embds = word_embds * mask.unsqueeze(-1)
    sentence_embd = self.pool(masked_word_embds.transpose(1,2)).transpose(1,2)
    residual_res = torch.relu(self.skip_layer(sentence_embd))
    residual_res = torch.relu(self.skip_layer2(residual_res)).squeeze()
    return residual_res

class Tree(object):
  def __init__(self, parent, mem_dim, key="", device="cpu", mem=None):
    self.parent = parent
    self.key = key
    self.memory = nn.Parameter(torch.randn(mem_dim)).to("cpu" if device == -1 else device) if mem == None else nn.Parameter(mem).to("cpu" if device == -1 else device)
    self.attention = 0
    self.children = torch.jit.annotate(List[Tree], [])
        
  def addChild(self, mem_dim, key="", device="cpu", mem=None):
    newTree = Tree(self, mem_dim, key, device, mem)
    self.children.append(newTree)
    return newTree
  
  

class EHRMemoryNetwork(nn.Module):
  def __init__(self, word_dim, mem_dim, hidden_dim, demo_dim):
    super(EHRMemoryNetwork, self).__init__()
    self.mem_dim = mem_dim

    self.demo_embd = ResidualBlock(demo_dim, hidden_dim, mem_dim)
    self.erase_layer = nn.Linear(word_dim, mem_dim)
    self.add_layer = nn.Linear(word_dim, mem_dim)
    
    self.init_mem = nn.Parameter(torch.randn(mem_dim))
    self.memory = None
    
  def forward(self, input, mask, labels, demo):    
    batch_size = input.size(0)
    time_step = input.size(1)
    modalities = 3
    depth = input.size(2) // modalities
    word_dim = input.size(3)
    
    self.memory = [Tree(None, self.mem_dim, "", input.get_device()) for _ in range(batch_size)]

    demo_embds = self.demo_embd(demo)
    for i in range(batch_size):
      self.memory[i].addChild(self.mem_dim, 'Demographics', input.get_device(), mem=demo_embds[i])

      for j in range(time_step):
        if mask[i,j] == 0:
          continue

        for k in range(modalities):
          if labels[i][j][k][0] == 'nan':
            continue

          cur_input = input[i, j, k*depth:(k+1)*depth, :].reshape(depth, word_dim)
          cur_tree = self.memory[i]
          for l in range(depth):
            curLabel = labels[i][j][k][l]
            match = None
            for tree in cur_tree.children:
              if tree.key == curLabel:
                match = tree
                break
            cur_tree = match if match != None else cur_tree.addChild(self.mem_dim, curLabel, input.get_device(), mem=self.init_mem)

            for m in range(l, depth):
              cur_step = cur_input[m]
              scale = 1 / (2 ** (m - l))
              erase = scale * torch.sigmoid(self.erase_layer(cur_step))
              add = scale * torch.tanh(self.add_layer(cur_step))
              cur_tree.memory = cur_tree.memory * (1 - erase) + add

    return self.memory



class QueryNetwork(nn.Module):
  def __init__(self, mem_dim, mlp_dim, output_dim=3):
    super(QueryNetwork, self).__init__()
    self.mlp = nn.Linear(2*mem_dim, mem_dim)
    self.mlp2 = nn.Linear(mem_dim, mlp_dim)
    self.output = nn.Linear(mlp_dim, output_dim)
    self.mem_dim = mem_dim
  
  def queryAll(self, mem, query, setTree=False):
    res = []
    att = []
    for i in range(len(mem)):
      child_trees = []
      child_trees.extend(mem[i].children)
      for t in child_trees:
        child_trees.extend(t.children)
      memory = [t.memory for t in child_trees]
      q = query[i].unsqueeze(0) # 1 * mem_dim
      memory = torch.stack(memory) # mem_len * mem_dim
      memory = memory.unsqueeze(0) # 1 * men_len * mem_dim
      attention = torch.bmm(q.unsqueeze(1), memory.permute(0,2,1)).squeeze(1) # 1 * mem_len
      attention = torch.softmax(attention, dim=-1) # 1 * mem_len
      response = attention.unsqueeze(-1) * memory # 1 * mem_len * mem_dim
      response = torch.sum(response, dim=1, keepdim=False).squeeze() # mem_dim
      if setTree:
        att_list = attention.squeeze().detach().cpu().tolist()
        if not isinstance(att_list, list):
          att_list = [att_list]
        for a, t in zip(att_list, child_trees):
          t.attention = a
      res.append(response)
      att.append(attention)
    return torch.stack(res), att

  def beamQuery(self, memory, query, beam, setTree):
    # memory: bs * tree
    # query: bs * mem_dim
    bs = len(memory)
    dev = query.device
    tree_lists = [[[c, -99999, False] for c in tree.children] for tree in memory]
    memory_lists = [[t[0].memory for t in l] for l in tree_lists]
    max_len = max([len(l) for l in tree_lists])
    memory_input = torch.zeros(bs, max_len, self.mem_dim).to(dev)
    for i in range(bs):
      memory_input[i,:len(memory_lists[i])] = torch.stack(memory_lists[i])
    attention = torch.bmm(query.unsqueeze(1), memory_input.permute(0,2,1)).squeeze(1) # bs * mem_len

    for i in range(bs):
      for j in range(len(tree_lists[i])):
        tree_lists[i][j][1] = attention[i][j]
      tree_lists[i].sort(key = lambda x: x[1], reverse=True)
      tree_lists[i] = tree_lists[i][0:beam]

    while False in [t[2] for l in tree_lists for t in l]:
      child_tree_lists = [[[c, -99999, False] for t in l for c in t[0].children if not t[2]] for l in tree_lists]
      memory_lists = [[t[0].memory for t in l] for l in child_tree_lists]
      max_len = max([len(l) for l in child_tree_lists])
      if max_len == 0:
        break

      memory_input = torch.zeros(bs, max_len, self.mem_dim).to(dev)
      for i in range(bs):
        if len(memory_lists[i]) == 0:
          continue
        memory_input[i,:len(memory_lists[i])] = torch.stack(memory_lists[i])
      attention = torch.bmm(query.unsqueeze(1), memory_input.permute(0,2,1)).squeeze(1) # bs * mem_len

      for i in range(bs):
        for j in range(len(tree_lists[i])):
          tree_lists[i][j][2] = True
        for j in range(len(child_tree_lists[i])):
          child_tree_lists[i][j][1] = attention[i][j]

        tree_lists[i] = tree_lists[i] + child_tree_lists[i]
        tree_lists[i].sort(key = lambda x: x[1], reverse=True)
        tree_lists[i] = tree_lists[i][0:beam]

    max_len = max([len(l) for l in tree_lists])
    memory_lists = [[t[0].memory for t in l] for l in tree_lists]
    memory_input = torch.zeros(bs, max_len, self.mem_dim).to(dev)
    for i in range(bs):
      memory_input[i,:len(memory_lists[i])] = torch.stack(memory_lists[i])

    attention = torch.bmm(query.unsqueeze(1), memory_input.permute(0,2,1)).squeeze(1) # bs * mem_len
    for i in range(bs):
      attention[i,len(tree_lists[i]):] = -99999
    attention = torch.softmax(attention, dim=-1) # bs * mem_len

    response = attention.unsqueeze(-1) * memory_input # bs * mem_len * mem_dim
    response = torch.sum(response, dim=1, keepdim=False) # mem_dim

    if setTree:
      for i in range(len(tree_lists)):
        for j in range(len(tree_lists[i])):
          tree_lists[i][j][0].attention = attention[i][j].cpu().detach()
        
    return response, attention

  def forward(self, memory, query, beam, getMemory=False):
    #query: bs, mem_dim
    #memory: bs, Tree, mem_dim
    responses, attentions = self.beamQuery(memory, query, beam, getMemory)
    output = torch.cat((responses, query), dim=-1)
    output = torch.relu(self.mlp(output))
    output = torch.relu(self.mlp2(output))
    output = self.output(output)
    output = output.squeeze()
    return output, responses, query, attentions, memory



def get_loss(criteria, criteria_mask, ehr, ehr_mask, demo, label, keys, query_network, ehr_network, criteria_network, beam, getMemory=False):
  criteria_embd = criteria_network(criteria, criteria_mask) # batch_size, mem_dim    
  memory = ehr_network(ehr, ehr_mask, keys, demo) # batch_size, Tree, mem_dim
  
  ce_loss = nn.CrossEntropyLoss()
  output, response, query, attention, memory = query_network(memory, criteria_embd, beam, getMemory)

  pred = torch.softmax(output, dim=-1)
  loss = ce_loss(output, label)

  return loss, pred, attention, response, query, memory