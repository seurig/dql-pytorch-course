import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(scores, n_avg, epsilons, filename):
    n = len(scores)
    x = np.arange(n)

    score_avgs = np.empty(shape=n)
    for idx in range(n):
        score_avgs[idx] = np.mean(scores[max(0, idx-n_avg):(idx+1)])

    print(len(score_avgs), len(epsilons))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)

    ax.plot(x, score_avgs, color='C0')
    ax.set_xlabel('game')
    ax.set_ylabel(f'{n_avg}-games moving average score', color='C0')

    ax2.plot(x, epsilons, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('epsilon', color='C1')
    ax2.yaxis.set_label_position('right')

    plt.savefig(filename)
    plt.close()