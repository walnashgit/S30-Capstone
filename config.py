IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100

projection_layer_config = dict(
    num_epoch=1,
    last_epoch=2,
    saved_model='', # model name with dir
    resume=False,
    data_dir='../data',
    checkpoint_dir='./checkpoint',
    max_seqlen=80,
    batch_size=2,
    vision_projector_file='',
    validation_phase=False,
    clip_dim=768, #512,
    phi_dim=2560,
    image_token='<image>'
)

multi_modal_config = dict(
    vision_projector_file='./checkpoint/projectionModel_ckpt_0.pth',
    checkpoint_dir='./checkpoint',
    finetuned_dir='./fine_tuned', #  '/Users/avinashkumaragarwal/PycharmProjects/pythonProject2/S29-QLORA-FineTuning/finetuned'
    resume=False,
    inference=False,
    raw_test=False,
    un_tuned=False,
    batch_size=16
)