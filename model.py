import os
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.metrics import categorical_accuracy


class CNN:
    def __init__(self,
                 input_shape=(784, ),
                 nb_classes=10,
                 optimizer=tf.train.AdamOptimizer(1e-3)):
        # create graph
        self.input_ = tf.placeholder(tf.float32, [None] + list(input_shape))
        self.feature_map, self.logit, self.output = self.build(self.input_, nb_classes)
        self.t = tf.placeholder(tf.float32, self.output.get_shape())
        self.loss = tf.reduce_mean(categorical_crossentropy(self.t, self.output))
        self.acc = tf.reduce_mean(categorical_accuracy(self.t, self.output))

        self.optimizer = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build(self, x, nb_classes):
        _x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        _x = Conv2D(16, (3, 3), activation='relu', padding='same')(_x)
        _x = MaxPool2D()(_x)
        _x = Conv2D(32, (3, 3), activation='relu', padding='same')(_x)
        _x = Conv2D(32, (3, 3), activation='relu', padding='same')(_x)
        _x = MaxPool2D()(_x)
        feature_map = _x
        # _x = GlobalAveragePooling2D()(_x)
        _x = Flatten()(_x)
        _x = Dense(512, activation='relu')(_x)
        logits = Dense(nb_classes, activation=None)(_x)
        outputs = Activation('softmax')(logits)
        return feature_map, logits, outputs

    def fit(self, data_generator, nb_epoch, model_dir):
        batch_size = data_generator.batch_size
        nb_sample = data_generator.n

        # calucuate steps per a epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1

        # fit loop
        for epoch in range(1, nb_epoch+1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            for step in range(steps_per_epoch):
                image_batch, label_batch = data_generator()
                _, loss, acc = self.sess.run([self.optimizer, self.loss, self.acc],
                                             feed_dict={self.input_: image_batch,
                                                        self.t: label_batch})
                print('{}/{}  loss : {:.4f}  acc : {:.4f}'.format(step, 
                                                                  steps_per_epoch, 
                                                                  loss, 
                                                                  acc),
                      end='\r')
        self.save(model_dir)
        print('\nTraining is done ... ')

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        self.saver.save(self.sess, os.path.join(model_dir, 'model.ckpt'))
        
    def evaluate_on_batch(self, x, y):
        return self.sess.run(self.acc,
                             feed_dict={self.input_: x,
                                        self.t: y})
    
    def evaluate_generator(self, data_generator):
        acc = 0
        for x, y in data_generator():
            acc += self.evaluate_on_batch(x, y) * len(x)
        return acc / data_generator.n
