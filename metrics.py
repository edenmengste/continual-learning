import numpy as np



def average_accuracy(results):

    """
    results:
    {
      task1:[acc after task1, acc after task2],
      task2:[acc after task2]
    }
    """


    final_scores=[]


    for task in results:

        final_scores.append(
            results[task][-1]
        )


    return np.mean(final_scores)




def forgetting_measure(results):

    """
    Forgetting = Best accuracy - Final accuracy
    """


    forgetting={}


    for task,accuracies in results.items():

        if len(accuracies)>1:

            forgetting[task] = (
                max(accuracies)
                -
                accuracies[-1]
            )

        else:

            forgetting[task]=0



    return forgetting