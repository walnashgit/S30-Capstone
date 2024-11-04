import torch
import torch.nn as nn

from S30ProjectionTraining.config import IGNORE_INDEX, IMAGE_TOKEN_INDEX


class PhiWithVision(torch.nn.Module):
    def __init__(self, phi_model, projectionModel, device='cuda', tokenizer=None):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.phi_model = phi_model
        self.projectionModel = projectionModel
        self.phi_embeddings = self.phi_model.get_input_embeddings()
        self.loss = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX, label_smoothing=0.1
        )

    def forward(self, input_ids=None, image_embedding=None, labels=None, **kwargs):
        projected_clip = self.projectionModel(image_embedding)
        combined_input, new_labels = self.prepare_input_embed(input_ids, projected_clip, labels)

        output = self.phi_model(
            inputs_embeds=combined_input
        )

        logits = output['logits']

        # pred_dict = generate_with_logits(logits)

        X = logits[:, :-1, :]
        Y = labels[:, 1:].contiguous().type(torch.LongTensor).to(self.device)

        X = X.contiguous().view(-1, X.size(-1))
        Y = Y.view(-1)

        loss_val = self.loss(
            X,
            Y
        )

        return dict(
            logits=logits,
            loss=loss_val
        )

    def prepare_input_embed(self, input_ids, projected_clip, labels):
        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_pos = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            image_token_start = image_pos[0]
            cur_new_input_embeds.append(self.phi_embeddings(cur_input_ids[:image_token_start]))
            cur_new_input_embeds.append(projected_clip[batch_idx])
            cur_new_input_embeds.append(self.phi_embeddings(cur_input_ids[image_token_start + 1:]))
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)

            new_input_embeds.append(cur_new_input_embeds)

            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = [cur_labels[:image_pos],
                                  torch.full((projected_clip[batch_idx].shape[0],), IGNORE_INDEX, device=labels.device,
                                             dtype=labels.dtype), cur_labels[image_pos + 1:]]

                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
                new_labels = torch.stack(new_labels, dim=0)

        new_input_embeds = torch.stack(new_input_embeds, dim=0)

        return new_input_embeds, new_labels

