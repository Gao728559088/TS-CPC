import matplotlib.pyplot as plt
import os

"""从日志中绘制loss和accuracy曲线"""

def parse_log_file(log_file_path):
    epochs = []
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []
    hyperparams = {}
    epoch = 1
    train_acc_sum = 0
    train_loss_sum = 0
    train_count = 0

    with open(log_file_path, 'r') as file:
        for line in file:
            if 'End epoch' in line:
                # 计算每个epoch的平均accuracy和loss
                if train_count > 0:
                    avg_train_acc = train_acc_sum / train_count
                    avg_train_loss = train_loss_sum / train_count
                    train_accuracies.append(avg_train_acc)
                    train_losses.append(avg_train_loss)
                epoch += 1
                train_acc_sum = 0
                train_loss_sum = 0
                train_count = 0
            elif 'Train Epoch:' in line:
                parts = line.split()
                train_acc_sum += float(parts[parts.index('Accuracy:') + 1])
                train_loss_sum += float(parts[parts.index('Loss:') + 1])
                train_count += 1
            elif 'Validation set:' in line:
                parts = line.split()
                val_loss = float(parts[parts.index('loss:') + 1])
                val_accuracy = float(parts[parts.index('Accuracy:') + 1])

                epochs.append(epoch)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
            elif any(param in line for param in ['log_interval', 'trajectory_window', 'timestep', 'masked_frames', 'n_warmup_steps', 'epochs', 'batch_size', 'seed', 'gpus']):
                param, value = line.strip().split(': ')
                hyperparams[param] = value
    
    return epochs, train_losses, train_accuracies, val_losses, val_accuracies, hyperparams

def plot_performance(epochs, train_losses, train_accuracies, val_losses, val_accuracies, hyperparams, log_file_path):
 
    plt.figure(figsize=(12, 6))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    # 找到最佳验证精度及其对应的epoch
    best_val_acc_epoch = val_accuracies.index(max(val_accuracies)) + 1
    best_val_acc = max(val_accuracies)
    best_val_loss = val_losses[best_val_acc_epoch - 1]

    # 在验证精度曲线上标记最佳点
    plt.annotate(f'Best Val Acc\nEpoch: {best_val_acc_epoch}\nAcc: {best_val_acc:.4f}\nLoss: {best_val_loss:.4f}', 
                 xy=(best_val_acc_epoch, best_val_acc), 
                 xytext=(best_val_acc_epoch, best_val_acc + 0.05),
                 arrowprops=dict(facecolor='green', shrink=0.05))

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    # Add hyperparameters text
    plt.figtext(0.33, 0.25, "\n".join([f"{k}: {v}" for k, v in hyperparams.items()]), ha="left", fontsize=8, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    # Save image
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(log_file_path), os.path.basename(log_file_path).replace('.log', '.png')))
    # plt.show()

# Function call
def drawFromLog(log_file_path):
    epochs, train_losses, train_accuracies, val_losses, val_accuracies, hyperparams = parse_log_file(log_file_path)
    plot_performance(epochs, train_losses, train_accuracies, val_losses, val_accuracies, hyperparams, log_file_path)

def main():
    drawFromLog('snapshot/cdc/cdc-2024-01-21_19_16_00.log')

if __name__ == '__main__':
    main()
