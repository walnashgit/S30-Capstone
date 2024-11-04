import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import itertools
from torch.nn.utils.rnn import pad_sequence
import gc

from S30ProjectionTraining.MultiModalWithVision import PhiWithVision
from S30ProjectionTraining.ProjectionLayer import ProjectionLayer
from S30ProjectionTraining.dataset import MultiModalLlavaDataset
from config import multi_modal_config as cfg


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "dense",
        "fc1",
        "fc2"
    ]
)


phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
phi_tokenizer.pad_token = phi_tokenizer.eos_token


def fine_tune(model, dataloader, num_epochs, device):
    # model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

    step_count = 0
    prev_loss = 1000
    bestLoss = 0
    bestStep = 0
    for epoch in range(num_epochs):
        total_loss = 0
        step_loss = 0
        batch_iterator = tqdm(dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
        # for batch in itertools.islice(batch_iterator, 2):
            optimizer.zero_grad()
            labels = batch['labels'].to(device)

            # with torch.autocast(device_type=device, dtype=torch.float16):

            input_ids = batch['input_ids'].to(device)
            image_embedding = batch['image_embedding'].to(device)

            # Forward pass
            logits, loss = model(input_ids, None, image_embedding, labels)

            total_loss += loss.item()
            step_loss += loss.item()

            loss.backward()

            # print(f"\n Epoch {epoch + 1}, step: {step_count}, Loss: {loss.item()}, total loss: {total_loss}")
            if loss.item() < prev_loss:
                bestLoss = loss.item()
                bestStep = step_count
                print(f"\n Epoch {epoch + 1}, step: {step_count}, Loss: {loss.item()}, total loss: {total_loss}")
                # save_model(epoch, model, loss.item(), optimizer, step_count)
                print('saving model')
                prev_loss = loss.item()
            elif step_count > 0 and step_count % 100 == 0:
                print(f"\n Epoch {epoch + 1}, step: {step_count}, loss: {loss.item()}, total loss: {total_loss}")

            if device == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()

            step_count += 1
            optimizer.step()

        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")
        print(f"BestLoss: {bestLoss}, BestStep: {bestStep}")


def save_model(epoch, model, loss, optimizer, step_count):
    print('saving model')
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'step_count': step_count,
    }, '%s/phiModel_ckpt_%s.pth' % (cfg['checkpoint_dir'], epoch))

    model.phi_model.save_pretrained(cfg['finetuned_dir'])
    phi_tokenizer.save_pretrained(cfg['finetuned_dir'])


def run_fine_tune():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'

    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(14)

    batch_size = 2
    dataset = MultiModalLlavaDataset('./coco2014_clip_embeddings_m2.h5',
                                     './llava_instruct_150k.json', phi_tokenizer)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

    model = get_model(device)
    model = model.to(device)

    fine_tune(model, dataloader, 1, device)

def get_model(device):
    projectionModel = ProjectionLayer(cfg['clip_dim'], cfg['phi_dim'])
    trained_proj_model = torch.load(cfg['vision_projector_file'])
    projectionModel.load_state_dict(trained_proj_model['model_state_dict'])
    # projectionModel.to(device)
    projectionModel.eval()
    for param in projectionModel.parameters():
        param.requires_grad = False

    phi_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
                                                     trust_remote_code=True,
                                                     quantization_config=bnb_config)
    phi_model.config.use_cache = False
    phi_model = prepare_model_for_kbit_training(phi_model)
    phi_model = get_peft_model(phi_model, peft_config)
    # phi_model.to(device)
    phi_model.train()

    return PhiWithVision(projectionModel, phi_model, device, phi_tokenizer)

if __name__ == '__main__':
    run_fine_tune()