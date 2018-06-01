import sys
import util
from neuron_network import NeuronNetwork

supported_mode = ['train', '5fold', 'test']
MINI_BATCH = 256

def validate_arguments(arguments):
    if len(arguments) < 4:
        print ('Missing arguments')
        return False
    if not (arguments[1] in supported_mode):
        print 'Invalid mode, supported modes are', supported_mode
        return False
    return True


if __name__ == "__main__":

    train_data = ''
    model_file = ''
    mode = ''
    if validate_arguments(sys.argv):
        train_data = sys.argv[3]
        model_file = sys.argv[2]
        mode = sys.argv[1]
    else:
        sys.exit("Invalid Arguments")
    inputs, outputs = util.load_test_data(train_data)

    if not inputs or not outputs:
        raise ValueError('Input data and output data cannot be empty')
        # exit(0)
    # inputs, outputs = util.load_test_data('Test_3')
    # inputs, outputs = util.load_test_data('UnitTest')
    nn = NeuronNetwork(inputs, outputs, 0.01)
    # set the number of node for input layer
    nn.il_node_num = 10
    # set the number of node for hidden layer 1
    nn.hl1_node_num = 10
    # set the number of node for hidden layer 2
    nn.hl2_node_num = 10
    # set the number of node for hidden layer 3
    nn.hl3_node_num = 10
    # set the number of node for out layer
    nn.ol_node_num = 1
    nn.batch_size = MINI_BATCH
    nn.epoch = 2
    nn.build_network()
    nn.start(mode, model_file)
