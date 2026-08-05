import torch
import torch.nn as nn

from ewc import ewc_penalty


def train_model(
        model,
        loader,
        optimizer,
        epochs=3,
        fisher=None,
        old_params=None,
        importance=1000
):

    criterion = nn.CrossEntropyLoss()


    model.train()


    for epoch in range(epochs):

        total_loss=0


        for images,labels in loader:


            optimizer.zero_grad()


            outputs=model(images)


            loss=criterion(
                outputs,
                labels
            )


            if fisher is not None and old_params is not None:

                # EWC penalty must be recomputed every batch since it
                # depends on the model's *current* parameter values
                loss = loss + ewc_penalty(
                    model,
                    old_params,
                    fisher,
                    importance
                )


            loss.backward()


            optimizer.step()


            total_loss+=loss.item()



        print(
            f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}"
        )



def evaluate(model,loader):

    model.eval()

    correct=0
    total=0


    with torch.no_grad():

        for images,labels in loader:

            outputs=model(images)


            predictions=torch.argmax(
                outputs,
                dim=1
            )


            correct += (
                predictions==labels
            ).sum().item()


            total += labels.size(0)



    return 100*correct/total