import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

def save_plots(train_acc_list, val_acc_list, train_loss_list, val_loss_list):
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc_list, color='green', lineStyle='-',
        label='train accuracy'
    )
    plot.plot(
        val_acc_list, color='blue', lineStyle='-',
        label='test accuracy'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('H:/seabird_family/1_DATASETS/by_image/train1.png')

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss_list, color='orange', lineStyles='-',
        label='train_loss'
    )

    plt.plot(
        val_loss_list, color='red', lineStyles='_',
        label="test_loss"
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('H:/seabird_family/1_DATASETS/by_image/val1.png')


