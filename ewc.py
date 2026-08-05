import torch
import torch.nn as nn



def calculate_fisher(model,loader):

    fisher={}


    for name,param in model.named_parameters():

        fisher[name]=torch.zeros_like(param)



    model.eval()


    criterion=nn.CrossEntropyLoss()



    for images,labels in loader:


        model.zero_grad()


        output=model(images)


        loss=criterion(
            output,
            labels
        )


        loss.backward()



        for name,param in model.named_parameters():

            fisher[name]+=param.grad.data**2



    for name in fisher:

        fisher[name]/=len(loader)



    return fisher




def ewc_penalty(
        model,
        old_parameters,
        fisher,
        importance=1000
):

    loss=0


    for name,param in model.named_parameters():

        loss += (
            fisher[name] *
            (
                param-old_parameters[name]
            )**2
        ).sum()



    return importance*loss