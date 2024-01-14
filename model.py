from transformers import AutoConfig,AutoModel
import torch
from gpa import GraphEncoder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from criterion import TripletLoss, Mine





class PLM_Graph(nn.Module):
    def __init__(self,config,num_labels,mod_type,graph,graph_type,layer,data_path,bce_wt,dot,tripmg=0,trip_penalty=0,mglist=None,
                 mine=0,mine_pen=0,netw='n1',min_proj=300,label_refiner=1,edge_dim=1):
        super(PLM_Graph, self).__init__()

        self.bert = AutoModel.from_pretrained(mod_type)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels=num_labels
        config.num_labels=num_labels
        self.bce_wt=bce_wt
        self.graph=graph
        self.dot=dot
        self.mine=mine
        if self.mine:
          self.mineloss=Mine(768,min_proj,netw)
          self.mine_pen=mine_pen
        self.tripmg=tripmg
        if self.tripmg:
          self.trp_pen=trip_penalty
          self.trploss=TripletLoss(mglist,data_path=data_path)

        if self.graph:
          self.gc1 = GraphEncoder(config, graph_type=graph_type, edge_dim=edge_dim,layer=layer, data_path=data_path,tokenizer=mod_type,label_refiner=label_refiner)
        self.classifier = nn.Linear(config.hidden_size, num_labels)


        
        
    def forward(self, input_ids, attention_mask,labels):
        bert_output = self.bert(input_ids, attention_mask)['last_hidden_state'][:, 0]
        bert_output = self.dropout(bert_output)
        if self.graph:


          label_embed = self.gc1(self.bert.embeddings)
          #label_embed = F.relu(label_embed)
          if self.dot:
            dot_product = torch.matmul(bert_output, label_embed.transpose(0,1))
            logits=dot_product

          else:   
            output = torch.zeros((bert_output.size(0), label_embed.size(0)), device=input_ids.device)
            for i in range(bert_output.size(0)):
                for j in range(label_embed.size(0)):
                    
                    output[i, j] = self.classifier(bert_output[i] + label_embed[j])[j]
            
            logits=output
        else:
          logits=self.classifier(bert_output)
        

        loss=0
        
        if self.training:
          if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            target = labels.to(torch.float32)
            loss += loss_fct(logits.view(-1, self.num_labels), target)*(self.bce_wt)

          if self.tripmg:
            #print(loss)
            #print('Inside Trip')
            loss+=(self.trploss(bert_output,label_embed,target)*self.trp_pen)
            #print(loss)
          if self.mine:
            loss+=(self.mineloss(bert_output,label_embed,target)*self.mine_pen)
    
        return {
            'loss': loss,
            'logits': logits,
            #'hidden_states': outputs.hidden_states,
            #'attentions': outputs.attentions,
            #'contrast_logits': contrast_logits,
            }
        
