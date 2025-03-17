
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import os



class MarginSeparationLoss(torch.nn.Module):
    def __init__(self, margin_list,data_path):
        super(MarginSeparationLoss, self).__init__()

        self.level_label_indices = torch.load(os.path.join(data_path,'level_dict.pt'))
        self.ms_losses = []

        for margin in margin_list:
            ms_losss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y), margin=margin)
            self.ms_losses.append(ms_losss)

        self.level_indices_list = [self.level_label_indices[level] for level in range(1, len(margin_list) + 1)]

    def forward(self, text_embeddings, label_embeddings, target_labels):
        #text_embeddings = text_embeddings[:, 0, :]
        label_list = [torch.nonzero(label).view(-1).tolist() for label in target_labels]

        text_embeddings, label_embeddings = F.normalize(text_embeddings, p=2, dim=-1), F.normalize(label_embeddings, p=2, dim=-1)
        similarity_matrix = F.cosine_similarity(text_embeddings.unsqueeze(1), label_embeddings.unsqueeze(0), dim=2)

        total_loss = 0

        for level, (level_indices, ms_loss) in enumerate(zip(self.level_indices_list, self.ms_losses)):
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

                total_loss += ms_loss(anchor_embed, positive_embed, negative_embed)
                
        #print(total_loss)
        return total_loss

