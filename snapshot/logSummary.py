import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后台

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
    lr_zero_epoch = None
    model_parameter = None

    lr_pattern = re.compile(r'lr:(\d+\.\d+)')
    param_pattern = re.compile(r'Model total parameter: (\d+)')

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
                lr_match = lr_pattern.search(line)
                if lr_match:
                    lr = float(lr_match.group(1))
                    if lr == 0.0 and lr_zero_epoch is None:
                        lr_zero_epoch = epoch
            elif 'Validation set:' in line:
                parts = line.split()
                val_loss = float(parts[parts.index('loss:') + 1])
                val_accuracy = float(parts[parts.index('Accuracy:') + 1])

                epochs.append(epoch)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
            elif 'Model total parameter:' in line:
                model_parameter = int(param_pattern.search(line).group(1))
            elif any(param in line for param in ['log_interval', 'trajectory_window', 'timestep', 'masked_frames', 'n_warmup_steps', 'epochs', 'batch_size', 'seed', 'gpus']):
                param, value = line.strip().split(': ')
                hyperparams[param] = value

    return epochs, train_losses, train_accuracies, val_losses, val_accuracies, hyperparams, lr_zero_epoch, model_parameter

def generate_log_summary(log_dir):
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    data = []

    for log_file in log_files:
        log_file_path = os.path.join(log_dir, log_file)
        epochs, train_losses, train_accuracies, val_losses, val_accuracies, hyperparams, lr_zero_epoch, model_parameter = parse_log_file(log_file_path)

        if val_accuracies:
            best_val_acc = max(val_accuracies)
            best_epoch = val_accuracies.index(best_val_acc) + 1
            best_val_loss = val_losses[best_epoch - 1]
        else:
            best_val_acc = None
            best_epoch = None
            best_val_loss = None

        params = {
            'ID': log_file[4:-4],  # 去除"cdc-"前缀和".log"后缀
            'trajectory_window': hyperparams.get('trajectory_window'),
            'batch_size': hyperparams.get('batch_size'),
            'n_warmup_steps': hyperparams.get('n_warmup_steps'),
            'timestep': hyperparams.get('timestep'),
            'epochs': hyperparams.get('epochs'),
            'best_val_accuracy': best_val_acc,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'lr=0@epoch': lr_zero_epoch,  # 添加lr=0@epoch列
            'parameter': model_parameter  # 添加模型参数列
        }
        data.append(params)
    
    df = pd.DataFrame(data)
    df = df[['ID', 'trajectory_window', 'batch_size', 'n_warmup_steps', 'timestep', 'epochs', 'best_val_accuracy', 'best_val_loss', 'best_epoch', 'lr=0@epoch', 'parameter']]
    df = df.sort_values(by='ID')
    
    return df.to_markdown(index=False)

def save_markdown_summary(log_dir, output_file='log_summary.md'):
    markdown_content = generate_log_summary(log_dir)
    with open(output_file, 'w') as md_file:
        md_file.write(markdown_content)
    print(f"Markdown summary generated: {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a markdown summary from log files.")
    parser.add_argument('--log_directory', type=str, default='snapshot/cdc/', help="Directory containing log files.")
    parser.add_argument('--output', type=str, default='snapshot/log_summary.md', help="Output markdown file name (default: log_summary.md).")

    args = parser.parse_args()
    save_markdown_summary(args.log_directory, args.output)
