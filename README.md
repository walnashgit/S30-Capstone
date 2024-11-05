# Multimodal GPT - Trained on PHI2 LLM

This project is a multimodal GPT capable of handling images, audio and text as input for context for question and answer from the GPT. This uses PHI2 LLM.

## Training

### Pre training
To handle image as input, a small projection layer was pre-trained which was used to project the input image embedding to the same latent space that PHI2 can handle.


* Image embedding is obtained using CLIP vision model.
* Image embedding is passed through projecting layer.
* Projected image embedding is concatenated with the embedding of the corresponding caption embedding got from the PHI2 text embedding. For this step the parameters of PHI2 is frozen so that it does change during back propagation.
* The concatenated embedding is used as input to PHI2 model and is compared with actual caption for loss. 
* Datset used is [COCO2014](https://www.kaggle.com/datasets/nadaibrahim/coco2014).


```
class ProjectionLayer(torch.nn.Module):
    def __init__(self, clip_dim, phi_dim):
        super().__init__()
        self.linear = torch.nn.Linear(clip_dim, phi_dim)

    def forward(self, x):
        return self.linear(x)
```

![image](https://github.com/user-attachments/assets/27978a2d-0b89-4e00-a486-4b82fee68008)


### Pre-Training Log

```
 Epoch 1, step: 4000, loss: 1.7207248210906982, total loss: 8806.965348124504
Processing Epoch 00:  26%|██▌       | 4101/15633 [1:37:46<4:29:48,  1.40s/it]

 Epoch 1, step: 4100, loss: 1.7700507640838623, total loss: 8986.26793396473
Processing Epoch 00:  27%|██▋       | 4201/15633 [1:40:06<4:23:33,  1.38s/it]

 Epoch 1, step: 4200, loss: 1.7114845514297485, total loss: 9165.797843456268
Processing Epoch 00:  28%|██▊       | 4301/15633 [1:42:28<4:22:25,  1.39s/it]

 Epoch 1, step: 4300, loss: 1.8178526163101196, total loss: 9349.606652259827
Processing Epoch 00:  28%|██▊       | 4401/15633 [1:44:48<4:29:18,  1.44s/it]

 Epoch 1, step: 4400, loss: 1.713474988937378, total loss: 9528.485793590546
Processing Epoch 00:  29%|██▉       | 4501/15633 [1:47:09<4:23:34,  1.42s/it]

 Epoch 1, step: 4500, loss: 1.8185430765151978, total loss: 9709.111825466156
Processing Epoch 00:  29%|██▉       | 4563/15633 [1:48:37<4:17:50,  1.40s/it]

 Epoch 1, step: 4562, Loss: 1.4059160947799683, total loss: 9814.87218272686
saving model
Processing Epoch 00:  29%|██▉       | 4602/15633 [1:49:31<4:15:43,  1.39s/it]

 Epoch 1, step: 4600, loss: 1.7384731769561768, total loss: 9877.717487335205
Processing Epoch 00:  30%|███       | 4701/15633 [1:51:51<4:15:05,  1.40s/it]

 Epoch 1, step: 4700, loss: 1.7126386165618896, total loss: 10048.258151650429
Processing Epoch 00:  31%|███       | 4777/15633 [1:53:38<4:23:39,  1.46s/it]

 Epoch 1, step: 4776, Loss: 1.3383914232254028, total loss: 10178.873227715492
saving model
Processing Epoch 00:  31%|███       | 4801/15633 [1:54:12<4:19:27,  1.44s/it]

 Epoch 1, step: 4800, loss: 1.7428756952285767, total loss: 10219.085227370262
Processing Epoch 00:  31%|███       | 4884/15633 [1:56:09<4:10:17,  1.40s/it]

 Epoch 1, step: 4883, Loss: 1.3373826742172241, total loss: 10357.566935658455
saving model
Processing Epoch 00:  31%|███▏      | 4901/15633 [1:56:33<4:16:20,  1.43s/it]

 Epoch 1, step: 4900, loss: 1.9052549600601196, total loss: 10385.680579423904
Processing Epoch 00:  32%|███▏      | 5001/15633 [1:58:53<4:06:28,  1.39s/it]

 Epoch 1, step: 5000, loss: 1.8309067487716675, total loss: 10549.668045520782
Processing Epoch 00:  32%|███▏      | 5026/15633 [1:59:29<4:11:36,  1.42s/it]

 Epoch 1, step: 5025, Loss: 1.2582365274429321, total loss: 10589.988594651222
saving model
Processing Epoch 00:  32%|███▏      | 5052/15633 [2:00:05<4:11:53,  1.43s/it]

 Epoch 1, step: 5051, Loss: 1.2571699619293213, total loss: 10630.821589589119
saving model
```



### Fine Tuning
After the projection layer was pretrained, PHI2 was fine tuned.

* Input image embedding is obtained using CLIP vision model.
* Image embedding is passed through pre-trained projecting layer. For fine tuning, the parameters of projection layer is forzen to prevent changes during back propagation.
* Projected embedding is concatenated with the embeddings of Question-Answer text data from [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) 
* Concatenated embedding is used as input to PHI2 model and the output is compared with the answer part for the loss.

![image](https://github.com/user-attachments/assets/015677c5-c610-4d97-b67b-18f1b40e0a77)


### Fine Tuning Log
```
 Epoch 1, step: 8580, loss: 2.147397041320801, total loss: 20604.322937846184
clearing cache
Processing Epoch 00:  47%|████▋     | 8591/18148 [7:30:17<8:25:17,  3.17s/it]

 Epoch 1, step: 8590, loss: 2.0401413440704346, total loss: 20627.469371914864
clearing cache
Processing Epoch 00:  47%|████▋     | 8601/18148 [7:30:48<8:24:41,  3.17s/it]

 Epoch 1, step: 8600, loss: 2.2512617111206055, total loss: 20651.634582281113
clearing cache
Processing Epoch 00:  47%|████▋     | 8611/18148 [7:31:19<8:25:37,  3.18s/it]

 Epoch 1, step: 8610, loss: 2.054892063140869, total loss: 20674.192289114
clearing cache
Processing Epoch 00:  48%|████▊     | 8621/18148 [7:31:50<8:22:25,  3.16s/it]

 Epoch 1, step: 8620, loss: 2.16609787940979, total loss: 20694.78110229969
clearing cache
Processing Epoch 00:  48%|████▊     | 8631/18148 [7:32:21<8:22:07,  3.17s/it]

 Epoch 1, step: 8630, loss: 2.1779747009277344, total loss: 20719.170233249664
clearing cache
Processing Epoch 00:  48%|████▊     | 8641/18148 [7:32:52<8:24:29,  3.18s/it]

 Epoch 1, step: 8640, loss: 2.852278709411621, total loss: 20743.28934788704
clearing cache
Processing Epoch 00:  48%|████▊     | 8651/18148 [7:33:23<8:22:59,  3.18s/it]

 Epoch 1, step: 8650, loss: 2.776125431060791, total loss: 20767.941595435143
clearing cache
Processing Epoch 00:  48%|████▊     | 8661/18148 [7:33:54<8:21:47,  3.17s/it]

 Epoch 1, step: 8660, loss: 2.1857335567474365, total loss: 20790.802763819695
clearing cache

```


## Further Improvements
* Fine tuned model is not accurate and can be trained further by improving the data input and loss calculation.
* The context of the queries is limited to the current query and can be improved to include the previous query and responses as context.
