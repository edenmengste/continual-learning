import torch
from torch.utils.data import DataLoader


from model import CNN
from train import train_model,evaluate

from replay import (
    create_memory,
    add_replay_data
)

from ewc import (
    calculate_fisher,
    ewc_penalty
)

from utils import save_parameters



def run_baseline(
        task1_loader,
        task2_loader,
        test1_loader,
        test2_loader
):


    print("\nBASELINE TRAINING")


    model=CNN()


    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


    results={

        "task1":[],
        "task2":[]

    }


    train_model(
        model,
        task1_loader,
        optimizer
    )


    results["task1"].append(
        evaluate(
            model,
            test1_loader
        )
    )



    train_model(
        model,
        task2_loader,
        optimizer
    )


    results["task1"].append(
        evaluate(
            model,
            test1_loader
        )
    )


    results["task2"].append(
        evaluate(
            model,
            test2_loader
        )
    )


    return results





def run_rehearsal(
        task1_loader,
        task2_loader,
        test1_loader,
        test2_loader
):


    print("\nREHEARSAL TRAINING")


    model=CNN()


    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


    memory=create_memory(
        task1_loader,
        500
    )


    train_model(
        model,
        task1_loader,
        optimizer
    )


    results={

        "task1":[
            evaluate(
                model,
                test1_loader
            )
        ],

        "task2":[]

    }


    model.train()


    for epoch in range(3):

        for images,labels in task2_loader:


            images,labels=add_replay_data(
                images,
                labels,
                memory
            )


            optimizer.zero_grad()


            output=model(images)


            loss=torch.nn.functional.cross_entropy(
                output,
                labels
            )


            loss.backward()

            optimizer.step()



    results["task1"].append(
        evaluate(
            model,
            test1_loader
        )
    )


    results["task2"].append(
        evaluate(
            model,
            test2_loader
        )
    )


    return results




def run_ewc(
        task1_loader,
        task2_loader,
        test1_loader,
        test2_loader,
        importance=1000
):


    print("\nEWC TRAINING")


    model=CNN()


    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


    train_model(
        model,
        task1_loader,
        optimizer
    )


    results={

        "task1":[
            evaluate(
                model,
                test1_loader
            )
        ],

        "task2":[]

    }


    # Snapshot how important each parameter was for task 1
    fisher = calculate_fisher(
        model,
        task1_loader
    )


    old_parameters = save_parameters(model)


    # Train on task 2 while penalizing drift on task-1-important weights
    train_model(
        model,
        task2_loader,
        optimizer,
        fisher=fisher,
        old_params=old_parameters,
        importance=importance
    )


    results["task1"].append(
        evaluate(
            model,
            test1_loader
        )
    )


    results["task2"].append(
        evaluate(
            model,
            test2_loader
        )
    )


    return results