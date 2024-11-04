# pretraining
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

from S30ProjectionTraining.ProjectionLayer import ProjectionLayer
from S30ProjectionTraining.ProjectionTrainer import ProjectionPreTrainer
from S30ProjectionTraining.dataset import ProjectionLayerDataset2
from config import projection_layer_config as cfg
from transformers import AutoModelForCausalLM, AutoTokenizer

from S30Capstone.LossFunction import CosineSimilarityLoss

phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
phi_tokenizer.pad_token = phi_tokenizer.eos_token
# phi_tokenizer.add_special_tokens({'additional_special_tokens': [cfg['image_token']]})


def pretrain_projection(model, dataloader, device):
    model = model.to(device)
    # optimizer = torch.optim.Adam(model.projectionModel.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    # criterion = CosineSimilarityLoss()
    # criterion = nn.CrossEntropyLoss()

    last_epoch = 0
    if cfg['resume']:
        saved_model = torch.load(cfg['saved_model'])
        model.projectionModel.load_state_dict(saved_model['model_state_dict'])
        optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        last_epoch = saved_model['last_epoch']

    step_count = -1
    num_epochs = cfg['num_epoch']
    # model.projectionModel.train()
    prev_loss = 10000
    bestLoss = 0
    bestStep = 0

    for epoch in range(last_epoch, num_epochs):
        total_loss = 0
        step_loss = 0
        # for batch in dataloader:
        batch_iterator = tqdm(dataloader, desc=f"Processing Epoch {epoch:02d}")
        # for batch in itertools.islice(batch_iterator, 20):
        for batch in batch_iterator:
            optimizer.zero_grad()
            # clip_embedding, texts = batch
            # clip_embedding = clip_embedding.to(device)
            # with torch.autocast(device_type=device, dtype=torch.float16):
            clip_embedding = batch['image_embedding']
            # ie_size = clip_embedding.size(1) - 1
            label_ids = batch['labels']
            prompt_ids = batch['prompt_ids']
            # image_token_position = batch['image_token_position']

            # projected_clip, phi_text_embedding = model(clip_embedding, text)
            # loss = criterion(projected_clip, phi_text_embedding)
            clip_embedding = clip_embedding.to(device).requires_grad_(True)
            label_ids = label_ids.to(device)
            prompt_ids = prompt_ids.to(device)
            # outputs = model(clip_embedding, label_ids, prompt_ids)
            logits, loss = model(clip_embedding, label_ids, prompt_ids)

            # if isinstance(outputs, tuple):
            #     logits = outputs[0]
            # elif hasattr(outputs, 'logits'):
            #     logits = outputs.logits
            # else:
            #     raise ValueError("Unexpected output format from the model")

            # X = logits[:, ie_size:ie_size + label_ids.size(1), :]
            # Y = label_ids.contiguous().type(torch.LongTensor).to(device)
            #
            # X = X.contiguous().view(-1, X.size(-1))
            # Y = Y.view(-1)
            #
            # loss = criterion(
            #     X,
            #     Y
            # )

            total_loss += loss.item()
            step_loss += loss.item()

            # print(f"\n Epoch {epoch + 1}, step: {step_count}, Loss: {loss.item()}, total loss: {total_loss}")

            # if step_count == -1:
            #     print(f"\n Epoch {epoch + 1}, step: {step_count}, Loss: {loss.item()}, total loss: {total_loss}")
            #
            # elif step_count > 0 and step_count % 100 == 0:
            # # elif step_count % 2 == 0:
            #     print(f"\n Epoch {epoch + 1}, step: {step_count}, Step loss: {step_loss/100}, total loss: {total_loss}")
            #     step_loss = 0
            #     # save_model(epoch, model, total_loss, optimizer, step_count)
            if loss.item() < prev_loss:
                bestLoss = loss.item()
                bestStep = step_count
                print(f"\n Epoch {epoch + 1}, step: {step_count}, Loss: {loss.item()}, total loss: {total_loss}")
                # save_model(epoch, model, total_loss, optimizer, step_count)
                print('saving model')
                prev_loss = loss.item()
            elif step_count > 0 and step_count % 100 == 0:
                print(f"\n Epoch {epoch + 1}, step: {step_count}, loss: {loss.item()}, total loss: {total_loss}")

            step_count += 1
            loss.backward()
            optimizer.step()

            gc.collect()
            torch.cuda.empty_cache()

        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")
        print(f"BestLoss: {bestLoss}, BestStep: {bestStep}")

    # return model.projectionModel


def save_model(epoch, model, loss, optimizer, step_count):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.projectionModel.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'step_count': step_count,
    }, '%s/projectionModel_ckpt_%s.pth' % (cfg['checkpoint_dir'], epoch))


def train():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(14)

    dataset = ProjectionLayerDataset2('./coco2014_clip_embeddings_m2.h5', './captions_train2014.json', phi_tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=False)

    model = get_model(device)
    pretrain_projection(model, dataloader, device)


def get_model(device):
    projectionModel = ProjectionLayer(cfg['clip_dim'], cfg['phi_dim'])
    phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    # phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True,
    #                                                  torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # phi_model = phi_model.half()

    projectionModel.train()
    phi_model.eval()  # Set Phi-2 to evaluation mode
    for param in phi_model.parameters():
        param.requires_grad = False

    return ProjectionPreTrainer(projectionModel, phi_model, phi_tokenizer, device)


def find_image_toke_pos(self, input_ids):
    # Find the position of the <image> token
    image_token = cfg['image_token']
    image_token_id = self.phi_tokenizer.convert_tokens_to_ids(image_token)
    image_token_position = (torch.tensor(input_ids) == image_token_id).nonzero(as_tuple=True)[0]
    return image_token_position


if __name__ == '__main__':
    train()
