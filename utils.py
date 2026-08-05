import torch



def save_parameters(model):

    parameters={}


    for name,param in model.named_parameters():

        parameters[name]=param.clone().detach()


    return parameters



def set_seed(seed=42):

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)