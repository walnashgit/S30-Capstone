import torch
import os
from peft import PeftModel
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, PreTrainedTokenizerFast
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer

from S30Capstone.ProjectionLayer import ProjectionLayer2, ProjectionLayer
from config import multi_modal_config as cfg

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# clip processor
clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# projection model
ckpt = torch.load(cfg['vision_projector_file'], map_location=torch.device(device))
if cfg['vision_projector_file'] == './checkpoint/vp_ckpt_0.pth':
    projectionModel = ProjectionLayer2(cfg['clip_dim'], cfg['phi_dim'])
else:
    projectionModel = ProjectionLayer(cfg['clip_dim'], cfg['phi_dim'])
projectionModel.load_state_dict(ckpt['model_state_dict'])

# PHI model
phi_base_model = AutoModelForCausalLM.from_pretrained(
    'microsoft/phi-2',
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float32,
    trust_remote_code=True
    # device_map=device_map,
)


# Tokeniser
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
tokenizer.pad_token = tokenizer.unk_token

#use this when <image> token was added.
# tokenizer_json_path = os.path.join(cfg['finetuned_dir'], 'tokenizer.json')
# tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json_path)
# tokenizer.pad_token = tokenizer.eos_token
# phi_base_model.resize_token_embeddings(len(tokenizer))

phi_new_model = cfg['finetuned_dir']
phi_model = PeftModel.from_pretrained(phi_base_model, phi_new_model)
phi_model = phi_model.merge_and_unload().to(device)


def infer(image_path, prompt):
    final_prompt = 'Human### ' + prompt + '\n' + 'AI### '
    prompt_embed = prompt_embedding(final_prompt)

    if image_path is not None:
        image_features = get_projected_image_embedding(image_path)
        inputs_embeds = torch.cat([image_features.to(device), prompt_embed], dim=1)
    else:
        inputs_embeds = prompt_embed

    out = phi_model.generate(inputs_embeds=inputs_embeds, min_new_tokens=10, max_new_tokens=50,
                             bos_token_id=tokenizer.bos_token_id)
    response = tokenizer.decode(out[0], skip_special_tokens=True)

    print(f"Generated text: {response}")


def get_projected_image_embedding(image_path):
    if image_path is not None:
        image = Image.open(image_path)
        clip_inputs = clip_processor(images=image, return_tensors="pt").to(device)
        # Add a dummy text input
        # clip_inputs['input_ids'] = torch.zeros((1, 1), dtype=torch.long, device=device)
        # clip_inputs['attention_mask'] = torch.ones((1, 1), dtype=torch.long, device=device)

        with torch.no_grad():
            if cfg['clip_dim'] == 768:
                x = clip_model(**clip_inputs, output_hidden_states=True)
                image_embedding = x.hidden_states[-4]#[:, 1:]
                # image_embedding = x.vision_model_output.hidden_states[-1] #.squeeze().cpu().numpy()
            else:
                image_embedding = clip_model.get_image_features(**clip_inputs)
        return projectionModel(image_embedding)


def prompt_embedding(prompt):
    prompt_ids = tokenizer.encode(prompt)
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.int32).unsqueeze(0).to(device)
    return phi_model.get_input_embeddings()(prompt_ids)


def test_projection_layer(image_path):
    image_features = get_projected_image_embedding(image_path)
    out = phi_base_model.generate(inputs_embeds=image_features, min_new_tokens=10, max_new_tokens=50,
                             bos_token_id=tokenizer.bos_token_id)
    response = tokenizer.decode(out[0], skip_special_tokens=True)

    print(f"projected text: {response}")


if __name__ == '__main__':
    # Example usage
    # image_path = None
    # image_path = "./COCO_train2014_000000000089.jpg"
    # image_path = "./COCO_train2014_000000000030.jpg"
    image_path = "./cat.jpg"
    # image_path = "./dog.jpg"
    # image_path = "./haanthi.jpg"
    # image_path = "./tabla04.jpg"
    # image_path = "./bna.jpg"

    text_prompt = "Explain the image?"
    # text_prompt = "which animal is in the image and what is it's color?"
    # text_prompt = "What is blackhole?"
    # text_prompt = "Is there a motorcycle in the image?"
    # text_prompt = "what is the color of the animal in the image?"

    infer(image_path, text_prompt)
    # test_projection_layer(image_path)