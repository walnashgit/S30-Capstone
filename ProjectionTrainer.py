import torch
import torch.nn as nn

from S30ProjectionTraining.config import IMAGE_TOKEN_INDEX


class ProjectionPreTrainer(torch.nn.Module):
    def __init__(self, projectionModel, phi_model, phi_tokenizer, device='cuda'):
        super().__init__()
        self.device = device
        self.projectionModel = projectionModel
        self.phi_model = phi_model
        self.phi_tokenizer = phi_tokenizer
        self.phi_model.to(device)
        self.phi_embeddings = self.phi_model.get_input_embeddings()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, image_embedding, label_ids, prompt_ids):
        projected_clip = self.projectionModel(image_embedding)
        # with torch.no_grad():
        new_prompt_embeds = self.prepare_input_embed(projected_clip, prompt_ids)
        ie_size = new_prompt_embeds.size(1) - 1

        phi_text_embedding = self.phi_embeddings(label_ids)
        combined_embeddings = torch.cat(
            [
                new_prompt_embeds,
                phi_text_embedding
            ],
            dim=1
        )
        phi_outputs = self.phi_model(inputs_embeds=combined_embeddings)

        logits = phi_outputs['logits']

        # pred_dict = generate_with_logits(logits[:, ie_size:ie_size + labels.size(1), :])

        X = logits[:, ie_size:ie_size + label_ids.size(1), :]
        Y = label_ids.contiguous().type(torch.LongTensor).to(self.device)

        X = X.contiguous().view(-1, X.size(-1))
        Y = Y.view(-1)

        loss_val = self.loss(
            X,
            Y
        )

        return logits, loss_val
        # return projected_clip, phi_text_embedding

    def prepare_input_embed(self, image_embeds, prompt_ids):
        new_input_embeds = []
        for batch_idx, cur_prompt_ids in enumerate(prompt_ids):
            image_token_indices = torch.where(cur_prompt_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []

            image_token_start = image_token_indices[0]

            cur_new_input_embeds.append(self.phi_embeddings(cur_prompt_ids[:image_token_start]))
            cur_new_input_embeds.append(image_embeds[batch_idx])
            cur_new_input_embeds.append(self.phi_embeddings(cur_prompt_ids[image_token_start + 1:]))
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)

            new_input_embeds.append(cur_new_input_embeds)

        new_input_embeds = torch.stack(new_input_embeds, dim=0)

        return new_input_embeds

    # def prepare_input_embed(self, prompt_ids, projected_clip, image_token_pos):
    #     new_input_embeds = []
    #     for batch_idx, cur_input_ids in enumerate(prompt_ids):
    #         pos = image_token_pos[batch_idx]
    #         phi_text_embedding = self.phi_embeddings(cur_input_ids)
    #         # if pos.numel() > 0 and pos.item() < seq_length: # lets assume image token pos is present. It should be
    #         cur_new_input_embeds = [phi_text_embedding[:pos], projected_clip[batch_idx], phi_text_embedding[pos + 1:]]
    #         cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
    #         new_input_embeds.append(cur_new_input_embeds)
    #
    #     new_input_embeds = torch.stack(new_input_embeds, dim=0)
    #     return new_input_embeds





