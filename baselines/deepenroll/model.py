#Pool and classification
import torch
from torch import nn
    
class ECEmbedding(nn.Module):
  def __init__(self, word_dim, align_dim):
    super(ECEmbedding, self).__init__()
    #Input: batch_size * sentence_len * embd_dim
    self.linear = nn.Linear(word_dim, align_dim)
    
  def forward(self, input, mask):
    #Input: B * L * embd_dim
    #Mask: B * L
    sent_embs = torch.relu(self.linear(input))
    return sent_embs
  
class EHREmbedding(nn.Module):
  def __init__(self, vocab_dim, demo_dim, align_dim):
    super(EHREmbedding, self).__init__()
    self.vocab_embd = nn.Embedding(vocab_dim, align_dim)
    self.diag_comb = nn.Linear(align_dim, align_dim)
    self.diag_visit = nn.Linear(align_dim, align_dim)
    self.demo_embd = nn.Linear(demo_dim, align_dim)
    self.final1 = nn.Linear(align_dim, align_dim)
    self.final2 = nn.Linear(align_dim, align_dim)
    
  def forward(self, input, demo, mask):
    combo_embd = (torch.relu(self.diag_comb(self.vocab_embd(input[:,:,0]))) * self.vocab_embd(input[:,:,1])) + (torch.relu(self.diag_comb(self.vocab_embd(input[:,:,0]))) * self.vocab_embd(input[:,:,2]))
    med_embd = torch.relu(self.diag_visit(self.vocab_embd(input[:,:,0]) + combo_embd))
    visit_embd = torch.relu(self.final1(med_embd) + self.final2(torch.relu(self.demo_embd(demo))).unsqueeze(1))
    return visit_embd
    

class AlignmentModule(nn.Module):
  def __init__(self, align_dim, mlp_dim):
    super(AlignmentModule, self).__init__()
    self.align_mapping = nn.Linear(align_dim, align_dim)
    self.r_mapping = nn.Linear(2*align_dim, align_dim)
    self.mlp = nn.Linear(4*align_dim, mlp_dim)
    self.output = nn.Linear(mlp_dim, 3)
  
  def forward(self, criteria, ehr, criteria_mask, ehr_mask):
    #criteria: bs, sent_len, align_dim
    #ehr: bs, pat_len, mem_dim
    #criteria_mask: bs, sent_len
    #ehr_mask: bs, pat_len
    alignment = torch.bmm(torch.relu(self.align_mapping(criteria)), torch.relu(self.align_mapping(ehr)).transpose(1,2)) # bs, sent_len, pat_len
    beta = torch.softmax((alignment + ((1 - criteria_mask) * -1e9).unsqueeze(-1)), dim=1)
    alpha = torch.softmax((alignment.transpose(1,2) + ((1 - ehr_mask) * -1e9).unsqueeze(-1)), dim=1).transpose(1,2)
    r1 = torch.sum(torch.relu(self.r_mapping(torch.cat(((alpha.unsqueeze(-1) * ehr.unsqueeze(1)).sum(2), criteria), dim=-1))), dim=1)
    r2 = torch.sum(torch.relu(self.r_mapping(torch.cat(((beta.unsqueeze(-1) * criteria.unsqueeze(2)).sum(1), ehr), dim=-1))), dim=1)
    m = torch.cat((r1, r2, r1*r2, r1-r2), dim=-1)
    output = self.mlp(m)
    output = torch.relu(output)
    output = self.output(output)
    return output

def get_loss(criteria, criteria_mask, 
             ehr, ehr_mask, demo, label,
             align_network, ehr_network, ec_network):
  
  ehr_embd = ehr_network(ehr, demo, ehr_mask) # batch_size, class_num
  criteria_embd = ec_network(criteria, criteria_mask) #ec_num, mem_dim
  ce_loss = nn.CrossEntropyLoss()
  
  output = align_network(criteria_embd, ehr_embd, criteria_mask, ehr_mask) #bs, 3
  pred = torch.softmax(output, dim=-1)
  loss = ce_loss(output, label)
  
  return loss, pred