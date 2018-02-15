import numpy as np
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras.utils import to_categorical


class DataGenerator(Iterator):
    def __init__(self, batch_size=64, shuffle=True, is_training=True, flatten=True):
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
        if is_training:
            self.x = train_x
            self.y = train_y
        else:
            self.x = test_x
            self.y = test_y
        self.is_training = is_training
        
        if flatten:
            # x: (N, 28, 28) → (N, 784)
            self.x = self.x.reshape(self.x.shape[0], self.x.shape[1]*self.x.shape[2])
        else:
            # x: (N, 28, 28) → (N, 28, 28, 1)
            self.x = np.expand_dims(self.x, axis=-1)
        self.x = self.x / 255
        # one-hot vectorize y
        self.y = to_categorical(self.y, 10)
        super().__init__(len(self.x), batch_size, shuffle, None)

    def __call__(self, *args, **kwargs):
        """
        if is_training: return (image_batch, label_batch)
        else          : return (generator yield  (image_batch, label_batch))
        """
        if self.is_training:
            return self._flow_on_training()
        else:
            return self._flow_on_test()

    def _flow_on_training(self):
        # get minibatch indices
        with self.lock:
            index_array = next(self.index_generator)
        image_batch = self.x[index_array]
        label_batch = self.y[index_array]
        return image_batch, label_batch

    def _flow_on_test(self):
        # create indices
        indexes = np.arange(self.n)

        # calculate steps per a test
        steps = self.n // self.batch_size
        if self.n % self.batch_size != 0:
            steps += 1
        
        # yield loop
        for i in range(steps):
            index_array = indexes[i*self.batch_size: (i+1)*self.batch_size]
            image_batch = self.x[index_array]
            label_batch = self.y[index_array]
            yield image_batch, label_batch
