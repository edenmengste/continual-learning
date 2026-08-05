from torch.utils.data import DataLoader


from data import (
    load_mnist,
    PermutedMNIST
)


from experiments import (
    run_baseline,
    run_rehearsal,
    run_ewc
)


from metrics import (
    average_accuracy,
    forgetting_measure
)


from utils import set_seed



set_seed()



train_dataset,test_dataset = load_mnist()



task1_train = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


task1_test = DataLoader(
    test_dataset,
    batch_size=64
)



task2_train = DataLoader(
    PermutedMNIST(train_dataset),
    batch_size=64,
    shuffle=True
)


task2_test = DataLoader(
    PermutedMNIST(test_dataset),
    batch_size=64
)


baseline_results = run_baseline(
    task1_train,
    task2_train,
    task1_test,
    task2_test
)


print("\nBaseline Results")
print(baseline_results)


print(
    "Average Accuracy:",
    average_accuracy(
        baseline_results
    )
)


print(
    "Forgetting:",
    forgetting_measure(
        baseline_results
    )
)




replay_results = run_rehearsal(
    task1_train,
    task2_train,
    task1_test,
    task2_test
)


print("\nReplay Results")
print(replay_results)


print(
    "Average Accuracy:",
    average_accuracy(
        replay_results
    )
)


print(
    "Forgetting:",
    forgetting_measure(
        replay_results
    )
)




ewc_results = run_ewc(
    task1_train,
    task2_train,
    task1_test,
    task2_test
)


print("\nEWC Results")
print(ewc_results)


print(
    "Average Accuracy:",
    average_accuracy(
        ewc_results
    )
)


print(
    "Forgetting:",
    forgetting_measure(
        ewc_results
    )
)




print("\n==============================")
print("SUMMARY")
print("==============================")

print(
    f"{'Method':<12}{'Avg Accuracy':<16}{'Forgetting (Task1)':<20}"
)

for name,results in [
    ("Baseline",baseline_results),
    ("Rehearsal",replay_results),
    ("EWC",ewc_results)
]:

    print(
        f"{name:<12}"
        f"{average_accuracy(results):<16.2f}"
        f"{forgetting_measure(results)['task1']:<20.2f}"
    )