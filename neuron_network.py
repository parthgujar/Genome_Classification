import numpy as np
import tensorflow as tf
from datetime import datetime
from datetime import timedelta
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuronNetwork(object):
    def __init__(self, x, y, epsilon):
        super(NeuronNetwork, self).__init__()
        # self.input_texts = tf.placeholder(tf.float32, [None, len(x)])
        self.__il_node_num = 5
        self.__hl1_node_num = 4
        self.__hl2_node_num = 3
        self.__hl3_node_num = 2
        self.__ol_node_num = 1
        self.__mini_batch_size = 1000
        self.__epoch = 1000
        self.__kfold = 5
        self.input_data = x  # Concrete input_holder data
        self.output_data = y  # Concrete output data
        self.__number_of_features = len(self.input_data[1])
        # print 'Number of input_data: ', self.number_of_features
        self.learning_rate = epsilon
        # Hold the input_data for training and testing
        self.input_holder = tf.placeholder(tf.float32,
                                           [None, self.number_of_features],
                                           name='input-holder')
        # Hold the output_data for training and testing
        self.y_holder = tf.placeholder(tf.float32, [None, self.label_count],
                                       name='output-holder')
        # Neuron network output_data
        self.nn_output = None
        self.test_accuracy_history = []
        self.train_accuracy_history = []
        self.items_trained = 0
        self.items_test = 0
        self.__confusion_matrix = {'TPR': 0, 'FPR': 0, 'FNR': 0, 'TNR': 0}

    def __define_weight_bias(self):
        """Define the weights and bias of layers
            weights are matrix hold random number between -b and b
            For now, b belongs to a set [-1, 1]
        """
        # Weights of the network is a matrix of 4 x max number of node in a
        # layer
        self.weights = {
            'w1': tf.Variable(tf.random_uniform(
                shape=(self.number_of_features, self.il_node_num), minval=-1.0,
                maxval=1.0, dtype=tf.float32), name='w1'),
            'w2': tf.Variable(
                tf.random_uniform(shape=(self.il_node_num, self.hl1_node_num),
                                  minval=-1.0, maxval=1.0, dtype=tf.float32),
                name='w2'),
            'w3': tf.Variable(
                tf.random_uniform(shape=(self.hl1_node_num, self.hl2_node_num),
                                  minval=-1.0, maxval=1.0, dtype=tf.float32),
                name='w3'),
            'w4': tf.Variable(
                tf.random_uniform(shape=(self.hl2_node_num, self.hl3_node_num),
                                  minval=-1.0, maxval=1.0, dtype=tf.float32),
                name='w4'),
            # because we have 6 sticky label
            'w5': tf.Variable(tf.truncated_normal(
                shape=(self.hl3_node_num, self.label_count), mean=0.0,
                stddev=1), name='w5')
        }

        self.bias = {
            'b1': tf.Variable(
                tf.random_uniform([self.il_node_num], minval=-1.0,
                                  maxval=1.0, dtype=tf.float32), name="b1"),
            'b2': tf.Variable(
                tf.random_uniform([self.hl1_node_num], minval=-1.0,
                                  maxval=1.0, dtype=tf.float32), name="b2"),
            'b3': tf.Variable(
                tf.random_uniform([self.hl2_node_num], minval=-1.0,
                                  maxval=1.0, dtype=tf.float32), name="b3"),
            'b4': tf.Variable(
                tf.random_uniform([self.hl3_node_num], minval=-1.0,
                                  maxval=1.0, dtype=tf.float32), name="b4"),
            # because we have 6 sticky label
            'b5': tf.Variable(
                tf.random_uniform([self.label_count], minval=-1.0,
                                  maxval=1, dtype=tf.float32), name='b5')
        }

    def build_network(self):
        self.__define_weight_bias()
        """ Define 5 layers for this homework.
            x is input_texts, w: weight, b: bias
        """
        input_layer = tf.add(tf.matmul(self.input_holder, self.weights['w1']),
                             self.bias['b1'])
        input_layer = tf.nn.relu(input_layer)

        hidden_layer_1 = tf.add(tf.matmul(input_layer, self.weights['w2']),
                                self.bias['b2'])
        hidden_layer_1 = tf.nn.relu(hidden_layer_1)

        hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, self.weights['w3']),
                                self.bias['b3'])
        hidden_layer_2 = tf.nn.relu(hidden_layer_2)

        hidden_layer_3 = tf.add(tf.matmul(hidden_layer_2, self.weights['w4']),
                                self.bias['b4'])
        hidden_layer_3 = tf.nn.relu(hidden_layer_3)

        output_layer = tf.matmul(hidden_layer_3, self.weights['w5']) + self.bias['b5']
        self.nn_output = output_layer
        return self.nn_output
        # return tf.nn.softmax(output_layer)

    # def start(self, mode, model_file.txt, train_data):
    def start(self, mode, model_file):
        # x, y = util.read_data(train_data)
        if mode == 'train':
            print 'Start training...'
            start_time = datetime.now()
            self.train(self.input_data, self.output_data, model_file)
            print 'Processing completed:\n',
            print self.items_trained, 'item(s) trained,'
            print 'Training time: ', datetime.now() - start_time
            print 'Training accuracy:', np.mean(self.train_accuracy_history)
        elif mode == '5fold':
            print 'Start cross-validation...'
            training_time, testing_time = self.cross_validate(
                self.input_data, self.output_data, self.kfold, model_file)
            print '5fold completed!!!'
            print self.items_trained, ' item(s) trained,'
            print self.items_test, ' item(s) tested'
            # print 'Training accuracy: ', np.mean(self.train_accuracy_history)
            print 'Testing accuracy: ', np.mean(self.test_accuracy_history)
            print 'Training time: ', training_time
            print 'Testing time: ', testing_time
        elif mode == 'test':
            print 'Start testing...'
            start_time = datetime.now()
            self.test(self.input_data, self.output_data, model_file)
            print 'Testing completed!'
            print self.items_test, ' item(s) tested'
            print 'Accuracy:', np.mean(self.test_accuracy_history)
            print 'Testing time: ', datetime.now() - start_time

    def train(self, train_x, train_y, model_file):
        """ Train the neural net and save the weights to model file
        """
        # Evaluate model
        self.items_trained = 0
        prediction = tf.nn.softmax(self.nn_output)
        # Backward propagation: update weights to minimize the cost using
        # cross_entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.nn_output, labels=self.y_holder))
        optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)
        # Count the number of training items
        items_count = 0
        # Mark the 1000 items step to print number of processed item
        grand = 1
        is_printed = False
        for e in range(self.epoch):
            # shuffle the data before training
            for i in range(0, len(train_x)):
                try:
                    j = random.randint(i + 1, len(train_x) - 1)
                    if i != j:
                        train_x[i], train_x[j] = train_x[j], train_x[i]
                        train_y[i], train_y[j] = train_y[j], train_y[i]
                except ValueError:
                    pass
                    # print 'End of the list when shuffling'

            # slice the training data into mini batches and train on
            # these batches
            for k in range(0, len(train_x), self.batch_size):
                batch_x = train_x[k:k + self.batch_size]
                batch_y = train_y[k:k + self.batch_size]
                session.run(optimizer,
                            feed_dict={self.input_holder: batch_x,
                                       self.y_holder: batch_y})

                correct_pred = tf.equal(tf.argmax(prediction, 1),
                                        tf.argmax(self.y_holder, 1))
                accuracy = tf.reduce_mean(
                    tf.cast(correct_pred, tf.float32))
                # lost, acc = session.run([cost, accuracy], feed_dict={
                #     self.input_holder: batch_x, self.y_holder: batch_y})
                # print 'Epoch ', e
                # print 'Cost / Accuracy:', lost, acc
                acc = session.run(accuracy, feed_dict=
                {self.input_holder: batch_x, self.y_holder: batch_y})

                self.items_trained += len(train_x)
                # print'Training accuracy:', accuracy
                self.train_accuracy_history.append(acc)
                items_count += len(batch_x)
                jp = items_count / 1000
                if items_count > (grand * 1000) and not is_printed:
                    print str(items_count - (
                        items_count % 1000)) + ' items processed'
                    is_printed = True
                    grand += 1
                elif jp >= grand:
                    is_printed = False


        # Save the optimized weights and the biases to the model file
        # print 'Save to file'
        saver = tf.train.Saver(
            [self.weights['w1'], self.weights['w2'], self.weights['w3'],
             self.weights['w4'], self.weights['w5'], self.bias['b1'],
             self.bias['b2'], self.bias['b3'], self.bias['b4'],
             self.bias['b5']])
        saver.save(session, model_file)
        # print 'W5-trained:\n', session.run(self.weights['w5'])
        # print 'b1-trained:\n', session.run(self.bias['b1'])
        session.close()
        self.items_trained = items_count

    def cross_validate(self, x, y, k, model_file):
        # Clear the history
        self.test_accuracy_history = []
        self.train_accuracy_history = []
        self.items_test = 0
        self.items_trained = 0
        prediction = tf.nn.softmax(self.nn_output)
        # Backward propagation: update weights to minimize the cost using
        # cross_entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.nn_output, labels=self.y_holder))
        optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)
        row = len(x)
        block = row / k
        # print 'Block:', block
        training_time = timedelta(0, 0, 0)
        testing_time = timedelta(0, 0, 0)
        items_count = 0
        # Mark the 1000 items step to print number of processed item
        grand = 1
        is_printed = False
        for i in range(k):
            # Pick the current Si as the subset for testing
            sl_i = slice(i * block, (i + 1) * block)
            text_x = x[sl_i]
            test_y = y[sl_i]
            # test_y = np.split(y, [i*k, (i + 1) * k], axis=0)
            # print 'Test Y:', i, test_y
            train_x = np.delete(x, np.s_[i * block: (i + 1) * block], axis=0)
            # print 'Train X:', i, train_x
            train_y = np.delete(y, np.s_[i * block: (i + 1) * block], axis=0)
            # print 'Train Y:', i, train_y
            print 'Training on Si except S[', i, ']'
            start_time = datetime.now()
            for e in range(self.epoch):
                # shuffle the data before training
                for r in range(0, len(train_x)):
                    try:
                        j = random.randint(r + 1, len(train_x) - 1)
                        if r != j:
                            train_x[r], train_x[j] = train_x[j], train_x[r]
                            train_y[r], train_y[j] = train_y[j], train_y[r]
                    except ValueError:
                        pass
                        # print 'End of the list when shuffling'

                # slice the training data into mini batches and train
                for b in range(0, len(train_x), self.batch_size):
                    batch_x = train_x[b:b + self.batch_size]
                    batch_y = train_y[b:b + self.batch_size]
                    session.run(optimizer,
                                feed_dict={self.input_holder: batch_x,
                                           self.y_holder: batch_y})

                    correct_pred = tf.equal(tf.argmax(prediction, 1),
                                            tf.argmax(self.y_holder, 1))
                    accuracy = tf.reduce_mean(
                        tf.cast(correct_pred, tf.float32))
                    accuracy = session.run(accuracy,
                                           feed_dict={
                                               self.input_holder: batch_x,
                                               self.y_holder: batch_y})
                    # The homework does not require to calculate the training
                    # accuracy
                    self.train_accuracy_history.append(accuracy)
                    items_count += len(batch_x)
                    jp = items_count / 1000
                    if items_count > (grand * 1000) and not is_printed:
                        print str(items_count - (
                            items_count % 1000)) + ' items processed'
                        is_printed = True
                        grand += 1
                    elif jp >= grand:
                        is_printed = False

            duration = datetime.now() - start_time
            training_time += duration
            print 'Testing on S[', i, '] data'
            start_time = datetime.now()
            correct_pred = tf.equal(tf.argmax(self.nn_output, 1),
                                    tf.argmax(self.y_holder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            accuracy = session.run(accuracy,
                                   feed_dict={self.input_holder: text_x,
                                              self.y_holder: test_y})
            self.items_test += len(text_x)
            self.test_accuracy_history.append(accuracy)

            duration = datetime.now() - start_time
            testing_time += duration
            # Save the weights at the last fold
            if i == k - 1:
                saver = tf.train.Saver([self.weights['w1'], self.weights['w2'],
                                        self.weights['w3'], self.weights['w4'],
                                        self.weights['w5']])
                saver.save(session, model_file)

        self.items_trained = items_count
        return training_time, testing_time

    def test(self, x, y, model_file):
        session = tf.Session()
        init = tf.global_variables_initializer()
        session.run(init)
        # Load model file which is saved from training step
        # saver = tf.train.import_meta_graph(model_file + '.meta')
        # saver.restore(session, tf.train.latest_checkpoint('./'))
        saver = tf.train.Saver([self.weights['w1'], self.weights['w2'],
                                self.weights['w3'], self.weights['w4'],
                                self.weights['w5'], self.bias['b1'],
                                self.bias['b2'], self.bias['b3'],
                                self.bias['b4'], self.bias['b5']])
        saver.restore(session, model_file)
        # print 'b1: \n', session.run('b1:0')

        # Assign the weights which are loaded from model file
        self.weights['w1'].assign(session.run('w1:0'))
        self.weights['w2'].assign(session.run('w2:0'))
        self.weights['w3'].assign(session.run('w3:0'))
        self.weights['w4'].assign(session.run('w4:0'))
        self.weights['w5'].assign(session.run('w5:0'))
        # print 'Weights:', session.run(self.weights['w5'])
        # Assign the biases which are loaded from model file
        self.bias['b1'].assign(session.run('b1:0'))
        self.bias['b2'].assign(session.run('b2:0'))
        self.bias['b3'].assign(session.run('b3:0'))
        self.bias['b4'].assign(session.run('b4:0'))
        self.bias['b5'].assign(session.run('b5:0'))
        # print 'b1-loaded:\n', session.run(self.bias['b1'])
        prediction = tf.nn.softmax(self.nn_output)

        correct_pred = tf.equal(tf.argmax(prediction, 1),
                                tf.argmax(self.y_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        acc_rate = session.run(accuracy, feed_dict={self.input_holder: x,
                                                    self.y_holder: y})
        self.items_test += len(x)
        self.test_accuracy_history.append(acc_rate)
        # print 'Inside test accuracy:', acc_rate
        # confusion_matrix_tf = tf.confusion_matrix(tf.argmax(self.nn_output, 1),
        #                                           tf.argmax(self.y_holder, 1))
        # # cm = confusion_matrix_tf.eval(feed_dict={self.input_holder: x,
        # # self.y_holder: y})
        # # cf = tf.confusion_matrix(labels=self.y_holder,
        # #                          predictions=correct_pred, num_classes=6)
        # print 'Confusion matrix:', session.run(cf,
        #                                        feed_dict={self.input_holder: x,
        #                                                   self.y_holder: y})
        session.close()

    @property
    def il_node_num(self):
        return self.__il_node_num

    @il_node_num.setter
    def il_node_num(self, value):
        if value <= 0:
            raise ValueError('Input layer: Number of node must greater than 0')
        else:
            self.__il_node_num = value

    @property
    def hl1_node_num(self):
        return self.__hl1_node_num

    @hl1_node_num.setter
    def hl1_node_num(self, value):
        if value <= 0:
            raise ValueError(
                'Hidden layer 1: Number of node must greater than 0')
        else:
            self.__hl1_node_num = value

    @property
    def hl2_node_num(self):
        return self.__hl2_node_num

    @hl2_node_num.setter
    def hl2_node_num(self, value):
        if value <= 0:
            raise ValueError(
                'Hidden layer 2: Number of node must greater than 0')
        else:
            self.__hl2_node_num = value

    @property
    def hl3_node_num(self):
        return self.__hl3_node_num

    @hl3_node_num.setter
    def hl3_node_num(self, value):
        if value <= 0:
            raise ValueError(
                'Hidden layer 3: Number of node must greater than 0')
        else:
            self.__hl3_node_num = value

    @property
    def ol_node_num(self):
        return self.__ol_node_num

    @ol_node_num.setter
    def ol_node_num(self, value):
        if value <= 0:
            raise ValueError(
                'Output layer: Number of node must greater than 0')
        else:
            self.__ol_node_num = value

    @property
    def batch_size(self):
        return self.__mini_batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value <= 0:
            raise ValueError('Batch size must greater than 0')
        else:
            self.__mini_batch_size = value

    @property
    def max_node_of_layers(self):
        return max(self.il_node_num, self.hl1_node_num, self.hl2_node_num,
                   self.hl3_node_num, self.ol_node_num)

    @property
    def epoch(self):
        return self.__epoch

    @epoch.setter
    def epoch(self, value):
        if value <= 0:
            raise ValueError('Epoch must greater than 0')
        else:
            self.__epoch = value

    @property
    def label_count(self):
        return 6

    @property
    def number_of_features(self):
        return self.__number_of_features

    @property
    def kfold(self):
        return self.__kfold

    @kfold.setter
    def kfold(self, value):
        if 0 < value <= 5:
            self.__kfold = value
        else:
            raise ValueError('Cross validation should be from 1 to 5')
