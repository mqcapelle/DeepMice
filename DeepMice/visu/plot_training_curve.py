import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_training_curve(train_acc, val_acc, test_acc,
                        title='GRU traning curve', fontsize=12,
                        file_name = 'GRU traning curve.png'):
  """ Line plot of prediction accuracy
  x-axis: number of epochs
  y-axis: accuracy
  """
  number_epoch = len(train_acc)
  acc_list = train_acc + val_acc + test_acc
  acc_type = ['Train accuracy']*number_epoch+['Validation accuracy']*number_epoch\
          +['Test accuracy']*number_epoch
  acc = {'Accuracy':acc_list, 'Type': acc_type,
         'Number of epochs': [i for i in range(number_epoch)]*3}
  Acc = pd.DataFrame(data=acc)

  plt.figure(figsize=(6, 4), dpi=300)
  sns.set_theme(style="darkgrid")
  sns.lineplot(data=Acc, x="Number of epochs", y="Accuracy", hue="Type", palette='Set2')

  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=fontsize, frameon=False)
  plt.xlim((0, number_epoch))
  plt.ylim((0,1))
  plt.title(title, fontsize=fontsize)

  plt.savefig(file_name, bbox_inches='tight')
