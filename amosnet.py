import tensorflow as tf
import numpy as np

class AmosNet(object):

    def __init__(self, x, keep_prob, num_classes, skip_layer,
                 weights_path='DEFAULT'):
        """Create the graph of the AlexNet model.

        Args:
            x: Placeholder for the input tensor.
            keep_prob: Dropout probability.
            num_classes: Number of classes in the dataset.
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code
        """
        # Parse input arguments into class variables
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'files/data1.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):

	    conv1 = conv(self.X, 11, 3, 96, name= 'conv1', strides=[1,4,4,1] ,padding='VALID', groups=1)
	    pool1 = pool(conv1, padding='VALID', name='pool1')
	    lrn1  = tf.nn.lrn(pool1, depth_radius=2, alpha=2e-5, beta=0.75,name='norm1')

	    conv2= conv(lrn1, 5, 96, 256, name= 'conv2', strides=[1,1,1,1] ,padding='SAME', groups=2)
	    pool2 = pool(conv2, padding='VALID', name='pool2')
	    lrn2  = tf.nn.lrn(pool2, depth_radius=2, alpha=2e-5, beta=0.75,name='norm2')

	    conv3 = conv(lrn2, 3, 256, 384, name='conv3', strides=[1,1,1,1] ,padding='SAME', groups=1)

	    conv4 = conv(conv3, 3, 384, 384, name='conv4', strides=[1,1,1,1] ,padding='SAME', groups=2)

	    conv5 = conv(conv4, 3, 384, 256, name='conv5', strides=[1,1,1,1] ,padding='SAME', groups=2)

	    conv6 = conv(conv5, 3, 256, 256, name='conv6', strides=[1,1,1,1] ,padding='SAME', groups=2)
	    pool6 = pool(conv6, padding='VALID', name='pool6')

	    fc7 = fc(pool6,  6*6*256, 4096, name='fc7_new', relu = 1)
	    self.fc8 = fc(fc7, 4096, self.NUM_CLASSES, name='fc8_new', relu = 0)

#AmosNet Conv-Layers
#     net_layers={}
#     net_layers['conv1'] = conv(x, 11, 3, 96, name= 'conv1', strides=[1,4,4,1] ,padding='VALID', groups=1)
#     net_layers['pool1'] = pool(net_layers['conv1'], padding='VALID', name='pool1')
#     net_layers['lrn1']  = tf.nn.lrn(net_layers['pool1'], depth_radius=2, alpha=2e-5, beta=0.75,name='norm1')

#     net_layers['conv2'] = conv(net_layers['lrn1'], 5, 96, 256, name= 'conv2', strides=[1,1,1,1] ,padding='SAME', groups=2)
#     net_layers['pool2'] = pool(net_layers['conv2'], padding='VALID', name='pool2')
#     net_layers['lrn2']  = tf.nn.lrn(net_layers['pool2'], depth_radius=2, alpha=2e-5, beta=0.75,name='norm2')

#     net_layers['conv3'] = conv(net_layers['lrn2'], 3, 256, 384, name='conv3', strides=[1,1,1,1] ,padding='SAME', groups=1)

#     net_layers['conv4'] = conv(net_layers['conv3'], 3, 384, 384, name='conv4', strides=[1,1,1,1] ,padding='SAME', groups=2)

#     net_layers['conv5'] = conv(net_layers['conv4'], 3, 384, 256, name='conv5', strides=[1,1,1,1] ,padding='SAME', groups=2)

#     net_layers['conv6'] = conv(net_layers['conv5'], 3, 256, 256, name='conv6', strides=[1,1,1,1] ,padding='SAME', groups=2)
#     net_layers['pool6'] = pool(net_layers['conv6'], padding='VALID', name='pool6')

#     net_layers['fc7'] = fc(net_layers['pool6'],  6*6*256, 4096, name='fc7_new', relu = 1)
#     net_layers['fc8'] = fc(net_layers['fc7'], 4096, 2543, name='fc8_new', relu = 0)

#     net_layers['prob'] = tf.nn.softmax(net_layers['fc8'])
#     net_layers['pred'] = tf.argmax(tf.nn.softmax(net_layers['fc8']), dimension = 1)

    #self.prob = tf.nn.softmax(fc8)
    #self.pred = tf.argmax(tf.nn.softmax(fc8), dimension = 1)


#         """Create the network graph."""
#         # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
#         conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
#         norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

#         # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
#         conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
#         pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
#         norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

#         # 3rd Layer: Conv (w ReLu)
#         conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

#         # 4th Layer: Conv (w ReLu) splitted into two groups
#         conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

#         # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
#         conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
#         pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

#         # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
#         flattened = tf.reshape(pool5, [-1, 6*6*256])
#         fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
#         dropout6 = dropout(fc6, self.KEEP_PROB)

#         # 7th Layer: FC (w ReLu) -> Dropout
#         fc7 = fc(dropout6, 4096, 4096, name='fc7')
#         dropout7 = dropout(fc7, self.KEEP_PROB)

#         # 8th Layer: FC and return unscaled activations
#         self.fc8 = fc(dropout7, 4096, self.NUM_CLASSES, relu=False, name='fc8')
    def weights_initalize(self, sess):
        pre_trained_weights = np.load(open(self.WEIGHTS_PATH, "rb"), encoding="latin1").item() #encoding
        keys = sorted(pre_trained_weights.keys())
        for k in keys:
		if k not in self.SKIP_LAYER:
		#for k in list(filter(lambda x: 'conv' in x,keys)):
		    with tf.variable_scope(k, reuse=True):
		        temp = tf.get_variable('weights')
		        sess.run(temp.assign(pre_trained_weights[k]['weights']))
		    with tf.variable_scope(k, reuse=True):
		        temp = tf.get_variable('biases')
			sess.run(temp.assign(pre_trained_weights[k]['biases']))

    def load_initial_weights(self, session):
        """Load weights from file into network.

        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        'biases') we need a special load function
        """
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))


def conv(input, filter_size, in_channels, out_channels, name, strides, padding, groups):
    with tf.variable_scope(name) as scope:
        filt = tf.get_variable('weights', shape=[filter_size, filter_size, int(in_channels/groups), out_channels])
        bias = tf.get_variable('biases',  shape=[out_channels])
    if groups == 1:
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input, filt, strides=strides, padding=padding), bias))
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(value=input,num_or_size_splits=groups,axis = 3)
        filt_groups = tf.split( value=filt,num_or_size_splits=groups,axis = 3)
        output_groups = [ tf.nn.conv2d( i, k, strides = strides, padding = padding) for i,k in zip(input_groups, filt_groups)]

        conv = tf.concat(values = output_groups,axis = 3,)
        return tf.nn.relu(tf.nn.bias_add(conv, bias))

def fc(input, in_channels, out_channels, name, relu):
    input = tf.reshape(input , [-1, in_channels])
    with tf.variable_scope(name) as scope:
        filt = tf.get_variable('weights', shape=[in_channels , out_channels])
        bias = tf.get_variable('biases',  shape=[out_channels])
    if relu:
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(input, filt), bias))
    else:
        return tf.nn.bias_add(tf.matmul(input, filt), bias)


def pool(input, padding, name):
    return tf.nn.max_pool(input, ksize=[1,3,3,1], strides=[1,2,2,1], padding=padding, name= name)
