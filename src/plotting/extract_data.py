import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.signal import savgol_filter

PATH_TO_EVENT_FILE = "./tensorboard/FedAvg/events.out.tfevents.1573661432.1ball.29879.0"
PATH_TO_EVENT_FILE2 = "./tensorboard/Baseline/events.out.tfevents.1573648044.1ball.3762.0"
PATH_TO_EVENT_FACENET = "./tensorboard/Facenet/events.out.tfevents.1575408137.1ball.9543.0"

tags = ["accuracy", "loss", "train_loss", "f1_score"]


def load_data(path_to_even_file: str, tag_name: str):
    values = []
    steps = []
    for e in tf.train.summary_iterator(path_to_even_file):
        for v in e.summary.value:
            if v.tag == tag_name:
                values.append(v.simple_value)
                steps.append(e.step)
    return np.array(steps), np.array(values)


def facenet():
    steps, values = load_data(PATH_TO_EVENT_FACENET, "tar")

    values2 = savgol_filter(values, 101, 3)
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.ylabel("TAR@FAR=0.001", fontsize=12)
    plt.xlabel("Liczba kroków", fontsize=12)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.plot(steps, values, "k", alpha=0.5)
    plt.plot(steps, values2, "k")

    plt.savefig("./facenet_train.pdf", format="pdf")
    plt.show()


def extract_data():
    steps_fed_avg, accuracy_fed_avg = load_data(PATH_TO_EVENT_FILE, "accuracy")
    steps_baseline, accuracy_baseline = load_data(PATH_TO_EVENT_FILE2, "accuracy")

    print(len(accuracy_fed_avg))
    print(len(accuracy_baseline))

    make_fed_avg_plot(steps_fed_avg, accuracy_fed_avg)
    plt.savefig("./fed_avg.pdf", format="pdf")
    plt.show()
    plt.figure()
    make_baseline_plot(steps_baseline, accuracy_baseline)
    plt.savefig("./baseline.pdf", format="pdf")
    plt.show()


def make_baseline_plot(x, y):
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Liczba kroków", fontsize=12)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    max_x = 50_000
    plt.plot(x, y, color="black")
    plt.hlines(0.80, 0, max_x, "k", "--", alpha=0.5)
    plt.text(330, .802, '80%', alpha=0.5)

    plt.hlines(0.82, 0, max_x, "k", "--", alpha=0.5)
    plt.text(330, .822, '82%', alpha=0.5)

    plt.xlim(0, max_x)
    plt.ylim(.60, .85)


def make_fed_avg_plot(x, y):
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Liczba kroków", fontsize=12)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    max_x = 1501
    plt.plot(x, y, color="black")
    plt.hlines(0.80, 0, max_x, "k", "--", alpha=0.5)
    plt.text(10, .802, '80%', alpha=0.5)

    plt.hlines(0.82, 0, max_x, "k", "--", alpha=0.5)
    plt.text(10, .822, '82%', alpha=0.5)

    plt.xlim(0, max_x)
    plt.ylim(.60, .85)


if __name__ == '__main__':
    facenet()
