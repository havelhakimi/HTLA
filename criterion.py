
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import os



class TripletLoss(torch.nn.Module):
    def __init__(self, margin_list,data_path):
        super(TripletLoss, self).__init__()

        self.level_label_indices = torch.load(os.path.join(data_path,'level.pt'))
        self.triplet_losses = []

        for margin in margin_list:
            triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=margin)
            self.triplet_losses.append(triplet_loss)

        self.level_indices_list = [self.level_label_indices[level] for level in range(1, len(margin_list) + 1)]

    def forward(self, text_embeddings, label_embeddings, target_labels):
        #text_embeddings = text_embeddings[:, 0, :]
        label_list = [torch.nonzero(label).view(-1).tolist() for label in target_labels]

        text_embeddings, label_embeddings = F.normalize(text_embeddings, p=2, dim=-1), F.normalize(label_embeddings, p=2, dim=-1)
        similarity_matrix = F.cosine_similarity(text_embeddings.unsqueeze(1), label_embeddings.unsqueeze(0), dim=2)

        total_loss = 0

        for level, (level_indices, triplet_loss) in enumerate(zip(self.level_indices_list, self.triplet_losses)):
            #print(level_indices)
            level_similarities_batch = torch.index_select(similarity_matrix, 1, torch.tensor(level_indices).to(text_embeddings.device))
            triplet_list = []

            index_to_label_map = {val: key for key, val in enumerate(level_indices)}

            for batch_idx, (label_id, level_similarities) in enumerate(zip(label_list, level_similarities_batch)):
                level_label_ids = [id for id in label_id if id in level_indices]

                for label_id in level_label_ids:
                    positive_index = label_id
                    positive_similarity = level_similarities[index_to_label_map[positive_index]]
                    indices = torch.nonzero(level_similarities > positive_similarity)

                    if len(indices) == 0:
                        negative_indices = [idx for idx in level_indices if idx != positive_index]
                        negative_index = random.choice(negative_indices)
                    else:
                        if len(indices) > 1:
                            max_index = torch.argmax(level_similarities[indices])
                            negative_index = indices[max_index].item()
                        else:
                            negative_index = indices[0].item()

                    triplet_list.append((text_embeddings[batch_idx], label_embeddings[positive_index], label_embeddings[negative_index]))

            if len(triplet_list) >= 3:
                anchor_embed, positive_embed, negative_embed = zip(*triplet_list)
                anchor_embed = torch.stack(anchor_embed)
                positive_embed = torch.stack(positive_embed)
                negative_embed = torch.stack(negative_embed)

                total_loss += triplet_loss(anchor_embed, positive_embed, negative_embed)

        return total_loss



class mine_n1(torch.nn.Module):
  def __init__(self,hid,trans):

    super(mine_n1,self).__init__()

    self.text_rep = nn.Linear(hid,trans)
    self.label_rep = nn.Linear(hid, trans)
    
    self.l0 = nn.Linear(trans+trans, trans)
    self.l1 = nn.Linear(trans, trans)
    self.l2 = nn.Linear(trans, 1)

  def forward(self,text_embed,label_embed):

    text_embed=self.text_rep(text_embed)

    label_embed=self.label_rep(label_embed)

    h = torch.cat((text_embed, label_embed), dim=1)

    h = F.relu(self.l0(h))
    h = F.relu(self.l1(h))

    return self.l2(h)


class mine_n2(torch.nn.Module):
  def __init__(self,hid,trans):

    super(mine_n2,self).__init__()

    self.text_rep = nn.Linear(hid,trans)
    self.label_rep = nn.Linear(hid, trans)
    
    self.l0 = nn.Linear(trans+trans, trans)
    #self.l1 = nn.Linear(trans, trans)
    self.l2 = nn.Linear(trans, 1)

  def forward(self,text_embed,label_embed):

    text_embed=self.text_rep(text_embed)

    label_embed=self.label_rep(label_embed)

    h = torch.cat((text_embed, label_embed), dim=1)

    h = F.relu(self.l0(h))
    #h = F.relu(self.l1(h))

    return self.l2(h)
    
class mine_n3(torch.nn.Module):
  def __init__(self,hid,trans):

    super(mine_n3,self).__init__()

    #self.text_rep = nn.Linear(hid,trans)
    #self.label_rep = nn.Linear(hid, trans)
    
    self.l0 = nn.Linear(768+768, 768)
    #self.l1 = nn.Linear(trans, trans)
    self.l2 = nn.Linear(768, 1)

  def forward(self,text_embed,label_embed):

    #text_embed=self.text_rep(text_embed)

    #label_embed=self.label_rep(label_embed)

    h = torch.cat((text_embed, label_embed), dim=1)

    h = F.relu(self.l0(h))
    #h = F.relu(self.l1(h))

    return self.l2(h)


class mine_n4(torch.nn.Module):
  def __init__(self,hid,trans):

    super(mine_n4,self).__init__()

    #self.text_rep = nn.Linear(hid,trans)
    #self.label_rep = nn.Linear(hid, trans)
    
    self.l0 = nn.Linear(768+768, 768)
    self.l1 = nn.Linear(768, 768)
    self.l2 = nn.Linear(768, 1)

  def forward(self,text_embed,label_embed):

    #text_embed=self.text_rep(text_embed)

    #label_embed=self.label_rep(label_embed)

    h = torch.cat((text_embed, label_embed), dim=1)

    h = F.relu(self.l0(h))
    h = F.relu(self.l1(h))

    return self.l2(h)


class mine_n5(torch.nn.Module):
  def __init__(self,hid,trans):

    super(mine_n5,self).__init__()

    #self.text_rep = nn.Linear(hid,trans)
    self.label_rep = nn.Linear(hid, trans)
    self.c0 = nn.Conv1d(768, 768, kernel_size=3)
    self.c1 = nn.Conv1d(768, trans, kernel_size=3)
    self.l0 = nn.Linear(trans+trans, trans)
    self.l1 = nn.Linear(trans, trans)
    self.l2 = nn.Linear(trans, 1)

  def forward(self,text_embed,label_embed):

    #text_embed=self.text_rep(text_embed)

    label_embed=self.label_rep(label_embed)
    text_embed = text_embed.permute(0, 2, 1)
    h = F.relu(self.c0(text_embed))
    h = self.c1(h)
    h = torch.mean(h, dim=2)
    h = h.view(label_embed.shape[0], -1)

    h = torch.cat((h, label_embed), dim=1)

    h = F.relu(self.l0(h))
    h = F.relu(self.l1(h))

    return self.l2(h)


class mine_n6(torch.nn.Module):
  def __init__(self,hid,trans):

    super(mine_n6,self).__init__()

    #self.text_rep = nn.Linear(hid,trans)
    self.label_rep = nn.Linear(hid, trans)
    #self.c0 = nn.Conv1d(768, 768, kernel_size=3)
    self.c1 = nn.Conv1d(768, trans, kernel_size=3)
    self.l0 = nn.Linear(trans+trans, trans)
    self.l1 = nn.Linear(trans, trans)
    self.l2 = nn.Linear(trans, 1)

  def forward(self,text_embed,label_embed):

    #text_embed=self.text_rep(text_embed)

    label_embed=self.label_rep(label_embed)
    text_embed = text_embed.permute(0, 2, 1)
    h = F.relu(self.c1(text_embed))
    #h = self.c1(h)
    h = torch.mean(h, dim=2)
    h = h.view(label_embed.shape[0], -1)

    h = torch.cat((h, label_embed), dim=1)
    
    h = F.relu(self.l0(h))
    h = F.relu(self.l1(h))

    return self.l2(h)



class Mine(torch.nn.Module):
  def __init__(self,hid,trans,typee):

    super(Mine,self).__init__()
    self.typee=typee
    if self.typee=='n1':
      self.network=mine_n1(hid,trans)
    if self.typee=='n2':
      self.network=mine_n2(hid,trans)
    
    if self.typee=='n3':
      self.network=mine_n3(hid,trans)

    if self.typee=='n4':
      self.network=mine_n4(hid,trans)
    
    if self.typee=='n5':
      self.network=mine_n5(hid,trans)
    
    if self.typee=='n6':
      self.network=mine_n6(hid,trans)
    


  def forward(self,text_embed,label_embed,target):

    if self.typee=='n5' or self.typee=='n6':
      pass
    
    else:
      pass
      #text_embed=text_embed[:,0,:]
    idx = np.random.permutation(text_embed.shape[0])
    negative_text = text_embed[idx,:]
    
    label_list=[ torch.nonzero(label).view(-1).tolist() for label in target]
   
    for i, label_index in enumerate(label_list):
            # Label Selector: select the corresponding labels for each text sample
      label_feature = label_embed[label_index,:]
      label_feature_mean = torch.mean(label_feature, dim=0, keepdim=True)
      if i == 0:
        label_feature_y = label_feature_mean
      else:
        label_feature_y = torch.cat((label_feature_y, label_feature_mean), dim=0)
                
      # compute the text-label mutual information maximization loss
      #t = text_feature.permute(0, 2, 1)
      #t_prime = negative_text.permute(0, 2, 1)
    E_joint = -F.softplus(-self.network(text_embed,label_feature_y)).mean()
    E_marginal = F.softplus(self.network(negative_text,label_feature_y)).mean()
    text_label_mi_disc_loss = (E_marginal - E_joint)

    #print(text_label_mi_disc_loss)
    return text_label_mi_disc_loss




class Tripletlosshard1(torch.nn.Module):

  def __init__(self,margin_list,data_path):
    super(Tripletlosshard1,self).__init__()

    self.level_dict= torch.load(os.path.join(data_path,'level.pt'))

    self.lvl_triploss=[]

    for margin in margin_list:

        self.lvl_triploss.append(nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=margin))

    self.lvl_idx=[]
    for level in range(1,len(margin_list)+1):
      self.lvl_idx.append(self.level_dict[level])
        
  
  def forward(self,text_embed,label_embed,target):


    
   
    #text_embed=text_embed[:,0,:]  
    label_list=[ torch.nonzero(label).view(-1).tolist() for label in target]

    text_embed,label_embed=F.normalize(text_embed,p=2,dim=-1),F.normalize(label_embed,p=2,dim=-1)
    sim = F.cosine_similarity(text_embed.unsqueeze(1), label_embed.unsqueeze(0), dim=2)

    
    lv1_sim_batch = torch.index_select(sim, 1, torch.tensor(self.lvl_idx[0]).to(text_embed.device))

    triplets_list = []
    map_={val:key for key,val in enumerate(self.lvl_idx[0])}
    inv_map={key:val for key,val in enumerate(self.lvl_idx[0])}
    #print(self.lv1_idx)
    for batch_idx,(label_id, lv1_sim) in enumerate(zip(label_list,lv1_sim_batch)):
      lv1_id=[id  for  id in label_id if id in self.lvl_idx[0]]
      #print(lv1_id)
      for id in lv1_id:

        pos_idx = id
        pos_sim=lv1_sim[map_[pos_idx]]
        indices = torch.nonzero(lv1_sim> pos_sim)

        if len(indices) == 0:
          neg_indices = [idx for idx in self.lvl_idx[0] if idx != pos_idx]
          max_idx_l1 = random.choice(neg_indices)
        

        else:
          if len(indices) > 1:
            max_index = torch.argmax(lv1_sim[indices])
            max_idx_l1 = indices[max_index]
            max_idx_l1=max_idx_l1.item()
            max_idx_l1=inv_map[max_idx_l1]

          else:

            max_idx_l1 = indices[0]
            max_idx_l1=max_idx_l1.item()  
            max_idx_l1=inv_map[max_idx_l1]


        #print(id)    
        triplets_list.append((text_embed[batch_idx], label_embed[pos_idx], label_embed[max_idx_l1]))

    # Concatenate tensors in the list into a new tensor
    anchor_embed, positive_embed, negative_embed = zip(*triplets_list)
    anchor_embed = torch.stack(anchor_embed)
    positive_embed = torch.stack(positive_embed)
    negative_embed = torch.stack(negative_embed)



    loss=self.lvl_triploss[0](anchor_embed,positive_embed,negative_embed)
    #print('Loss at level 1')
    
    
    # level 2
    #print('#######At level 2')
    lv1_sim_batch = torch.index_select(sim, 1, torch.tensor(self.lvl_idx[1]).to(text_embed.device))
    triplets_list = []

    map_={val:key for key,val in enumerate(self.lvl_idx[1])}
    inv_map={key:val for key,val in enumerate(self.lvl_idx[1])}
    for batch_idx,(label_id, lv1_sim) in enumerate(zip(label_list,lv1_sim_batch)):
      lv1_id=[id  for  id in label_id if id in self.lvl_idx[1]]
      #print(lv1_id)
      for id in lv1_id:
        pos_idx = id
        pos_sim=lv1_sim[map_[pos_idx]]
        indices = torch.nonzero(lv1_sim> pos_sim)

        if len(indices) == 0:
          neg_indices = [idx for idx in self.lvl_idx[1] if idx != pos_idx]
          max_idx_l1 = random.choice(neg_indices)
        

        else:
          if len(indices) > 1:
            max_index = torch.argmax(lv1_sim[indices])
            max_idx_l1 = indices[max_index]
            max_idx_l1=max_idx_l1.item()
            max_idx_l1=inv_map[max_idx_l1]

          else:

            max_idx_l1 = indices[0]
            max_idx_l1=max_idx_l1.item()  
            max_idx_l1=inv_map[max_idx_l1]

        triplets_list.append((text_embed[batch_idx], label_embed[pos_idx], label_embed[max_idx_l1]))
        

    if len(triplets_list) >= 3:    
      anchor_embed, positive_embed, negative_embed = zip(*triplets_list)
   
      anchor_embed = torch.stack(anchor_embed)
      positive_embed = torch.stack(positive_embed)
      negative_embed = torch.stack(negative_embed)        

    
      loss+=self.lvl_triploss[1](anchor_embed,positive_embed,negative_embed)

    return loss




class Tripletlosshard2(torch.nn.Module):

    def __init__(self, margin_list, data_path):
        super(Tripletlosshard2, self).__init__()

        self.level_dict = torch.load(os.path.join(data_path, 'level.pt'))

        self.lvl_triploss = []
        for margin in margin_list:
            self.lvl_triploss.append(nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=margin))

        self.lvl_idx = []
        for level in range(1, len(margin_list) + 1):
            self.lvl_idx.append(self.level_dict[level])

    def forward(self, text_embed, label_embed, target):
        #text_embed = text_embed[:, 0, :]

        label_list = [torch.nonzero(label).view(-1).tolist() for label in target]
        text_embed, label_embed = F.normalize(text_embed, p=2, dim=-1), F.normalize(label_embed, p=2, dim=-1)
        sim = F.cosine_similarity(text_embed.unsqueeze(1), label_embed.unsqueeze(0), dim=2)

        total_loss = 0.0
        for level, lvl_idx in enumerate(self.lvl_idx):
            lvl_sim_batch = torch.index_select(sim, 1, torch.tensor(lvl_idx).to(text_embed.device))
            triplets_list = []
            map_ = {val: key for key, val in enumerate(lvl_idx)}
            inv_map = {key: val for key, val in enumerate(lvl_idx)}

            for batch_idx, (label_id, lvl_sim) in enumerate(zip(label_list, lvl_sim_batch)):
                lvl_id = [id for id in label_id if id in lvl_idx]
                for id in lvl_id:
                    pos_idx = id
                    pos_sim = lvl_sim[map_[pos_idx]]
                    indices = torch.nonzero(lvl_sim > pos_sim)

                    if len(indices) == 0:
                        neg_indices = [idx for idx in lvl_idx if idx != pos_idx]
                        max_idx_l1 = random.choice(neg_indices)
                    else:
                        if len(indices) > 1:
                            max_index = torch.argmax(lvl_sim[indices])
                            max_idx_l1 = indices[max_index]
                            max_idx_l1 = max_idx_l1.item()
                            max_idx_l1 = inv_map[max_idx_l1]
                        else:
                            max_idx_l1 = indices[0]
                            max_idx_l1 = max_idx_l1.item()
                            max_idx_l1 = inv_map[max_idx_l1]

                    triplets_list.append((text_embed[batch_idx], label_embed[pos_idx], label_embed[max_idx_l1]))

            if len(triplets_list) >= 3:
                anchor_embed, positive_embed, negative_embed = zip(*triplets_list)
                anchor_embed = torch.stack(anchor_embed)
                positive_embed = torch.stack(positive_embed)
                negative_embed = torch.stack(negative_embed)

                total_loss += self.lvl_triploss[level](anchor_embed, positive_embed, negative_embed)

        return total_loss
