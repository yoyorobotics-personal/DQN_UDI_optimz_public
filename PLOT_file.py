import matplotlib.pyplot as plt
import torch
from CONFI import*

# This function will plot the mean values of each episode
def plot_means(mean_values, show_result=False):
    plt.figure(1)
    means_t = torch.tensor(mean_values, dtype=torch.float)
    if show_result:
        plt.title('Final Results')
    else:
        plt.clf()
        plt.title('Training Progress')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.plot(means_t.numpy(), label='Value per Episode')

    # Calculate and plot a moving average over the last 100 episodes
    if len(means_t) >= PLOT_AVE:
        moving_avg = means_t.unfold(0, PLOT_AVE, 1).mean(1).view(-1)
        moving_avg = torch.cat((torch.zeros(PLOT_AVE-1), moving_avg))
        plt.plot(moving_avg.numpy(), label='Episode Moving Average')

    plt.legend()
    plt.pause(0.001)  # Pause to update the plot in real-time

    '''
    # Display updates if running in Jupyter Notebook (optional)
    if 'inline' in plt.get_backend():
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
    '''