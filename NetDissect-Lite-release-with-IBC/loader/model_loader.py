import settings
import torch
import torchvision
import my_net


# used for main
def load_model(hook_fn):
    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        if settings.CUSTOM:
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if 'enc_style' not in k:
                    del state_dict[k]
            new_state_dict = {str.replace(k, 'enc_style.', ''): v for k, v in state_dict.items()}
            model = my_net.AlexEncoderDecouple(fea_dim=settings.CODE_LEN, class_num=settings.NUM_CLASSES)
            model.load_state_dict(new_state_dict)
        else:
            if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
                model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
                if settings.MODEL_PARALLEL:  # multi-gpu trained
                    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                        'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
            else:
                model = checkpoint
            for name in settings.FEATURE_NAMES:
                model._modules.get(name).register_forward_hook(hook_fn)  # hook feature stored in the list features_blobs

    if settings.GPU:
        model.cuda()
    model.eval()
    return model
