import torch


def create_memory(loader,memory_size=500):

    memory=[]


    for images,labels in loader:

        for i in range(len(images)):


            memory.append(
                (
                    images[i],
                    labels[i]
                )
            )


            if len(memory)>=memory_size:

                return memory


    return memory



def add_replay_data(
        images,
        labels,
        memory
):

    if len(memory)==0:

        return images,labels


    replay_samples = memory[:32]


    old_images,old_labels = zip(
        *replay_samples
    )


    old_images=torch.stack(old_images)

    old_labels=torch.tensor(old_labels)



    images=torch.cat(
        [
            images,
            old_images
        ]
    )


    labels=torch.cat(
        [
            labels,
            old_labels
        ]
    )


    return images,labels