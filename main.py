from datasets.MNIST import MNIST_dataset


def main():

    MNIST = MNIST_dataset(batch_size_train=128, batch_size_test=1000)
    MNIST.import_data()
    MNIST.train()
    MNIST.result()


if __name__ == "__main__":
    main()
