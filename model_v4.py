import numpy as np
import os
from sklearn.metrics.pairwise import pairwise_distances
from argparse import ArgumentParser

from tensorflow.keras.layers import Input, Dense, \
                                    Conv2D, MaxPooling2D, \
                                    AveragePooling2D, Dropout, \
                                    BatchNormalization, Flatten, \
                                    Concatenate, GlobalAveragePooling2D, \
                                    Lambda, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.backend import l2_normalize


class FaceEncoder():
    def __init__(self):

        self.order_w_name_idx = (1, 2, 3, 0)
        self.order_b_name_idx = (1, 0)

        input_size = Input(shape=(96, 96, 3))

        self.conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2),
                       padding='same',  kernel_regularizer=l2(0.01),
                       name='conv1')(input_size)
        self.bn1 = BatchNormalization(name='bn1')(self.conv1)
        act = Activation('relu')(self.bn1)

        mp = MaxPooling2D(pool_size=(3, 3),
                             strides=(2, 2),
                             padding='same')(act)

        self.conv2 = Conv2D(filters=64, kernel_size=(1, 1),
                       padding='same',  kernel_regularizer=l2(0.01),
                       name='conv2')(mp)
        self.bn2 = BatchNormalization(name='bn2')(self.conv2)
        act = Activation('relu')(self.bn2)


        self.conv3 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1),
                            padding='same',  kernel_regularizer=l2(0.01),
                            name='conv3')(act)
        self.bn3 = BatchNormalization(name='bn3')(self.conv3)
        act = Activation('relu')(self.bn3)

        mp = MaxPooling2D(pool_size=(3, 3),
                          strides=(2, 2),
                          padding='same')(act)

        # Inception blocks
        self.inception_3a = self.create_interception_block(mp, addition_name='3a',
                                                           conv1_5x5=16, conv2_5x5=32,
                                                           conv1_3x3=96, conv2_3x3=128,
                                                           conv_filters=64, pool_filters=32,
                                                           pool_type='max')

        self.inception_3b = self.create_interception_block(self.inception_3a, addition_name='3b',
                                                           conv1_5x5=32, conv2_5x5=64,
                                                           conv1_3x3=96, conv2_3x3=128,
                                                           conv_filters=64, pool_filters=64,
                                                           pool_type='avg')

        self.inception_3c = self.create_interception_block(self.inception_3b, addition_name='3c',
                                                           conv1_5x5=32, conv2_5x5=64,
                                                           conv1_3x3=128, conv2_3x3=256,
                                                           conv_filters=None, pool_filters=None,
                                                           pool_type='max', addition_pool=True)

        self.inception_4a = self.create_interception_block(self.inception_3c, addition_name='4a',
                                                           conv1_5x5=32, conv2_5x5=64,
                                                           conv1_3x3=96, conv2_3x3=192,
                                                           conv_filters=256, pool_filters=128,
                                                           pool_type='avg')
        self.inception_4e = self.create_interception_block(self.inception_4a, addition_name='4e',
                                                           conv1_5x5=64, conv2_5x5=128,
                                                           conv1_3x3=160, conv2_3x3=256,
                                                           conv_filters=None, pool_filters=None,
                                                           pool_type='max', addition_pool=True)

        self.inception_5a = self.create_interception_block(self.inception_4e, addition_name='5a',
                                                           conv1_5x5=None, conv2_5x5=None,
                                                           conv1_3x3=96, conv2_3x3=384,
                                                           conv_filters=256, pool_filters=96,
                                                           pool_type='avg')
        self.inception_5b = self.create_interception_block(self.inception_5a, addition_name='5b',
                                                           conv1_5x5=None, conv2_5x5=None,
                                                           conv1_3x3=96, conv2_3x3=384,
                                                           conv_filters=256, pool_filters=96,
                                                           pool_type='max')

        # Common last layers
        avg = GlobalAveragePooling2D(name='avg_pool')(self.inception_5b)
        self.flatten_layer = Flatten()(avg)
        self.dense_layer = Dense(units=128, kernel_regularizer=l2(0.01))(self.flatten_layer)
        self.out_layer = Lambda(lambda x: l2_normalize(x, axis=1))(self.dense_layer)

        model = Model(input_size, self.out_layer)
        model.summary()

        self.model = model

    @staticmethod
    def get_all_weights(root_weights_path):
        """
        Create dict filename: weights_values
        :param root_weights_path: path to folder with weights
        :return: dictionary {filename: weights_values}
        """
        full_list = os.listdir(root_weights_path)
        # TODO: regular expresions
        layer_name_selector = lambda filename: filename.split('.')[0]
        weight_loader = lambda filename: np.loadtxt(os.path.join(root_weights_path, filename), delimiter=',',
                                                                                               dtype=np.float32)
        model_weights_by_layers = {layer_name_selector(filename): weight_loader(filename) for filename in full_list}
        return model_weights_by_layers

    def load_weights(self, root_weights_path):
        """
        Load weights from filer to model
        :param root_weights_path:  path to folder with weights
        :return: None
        """
        model_weights_by_layers = self.get_all_weights(root_weights_path)
        grouped_weights = {}
        for x in model_weights_by_layers:
            main_name = x[:-2]
            if main_name in grouped_weights:
                grouped_weights[main_name].append(x)
            else:
                grouped_weights[main_name] = [x]

        for layer_name, files_names in grouped_weights.items():
           files_names = sorted(files_names) #bw/bmvw
           order_set = self.order_w_name_idx if len(files_names) == 4 else self.order_b_name_idx
           files_names = [x for _, x in sorted(zip(order_set, files_names), key=lambda pair: pair[0])]

           try:
                layer = self.model.get_layer(name=layer_name)
                src_weights = layer.get_weights()
                weights_list = [model_weights_by_layers[filename] for filename in files_names]
                weights_list[0] = weights_list[0].reshape(src_weights[0].shape)
                layer.set_weights(weights_list)
           except (AttributeError, ValueError) as e:
               print(e)

    def create_conv_pool_layers_set(self, input_layer, addition_name='', conv_filters=64, pool_filters=32, pool_type='max'):
        """
        Create 2 additional streams of layers:
                                            input_layer -> pooling-> conv_1
                                            input_layer -> conv_2
        :param input_layer: the last layer
        :param addition_name: index in the naming system (idx of inception layer)
        :param conv_filters: num of filters in conv_2 layer
        :param pool_filters: num of filters in conv_1 layer
        :param pool_type: 'max' - use MaxPooling or 'avg' - use AveragePooling
        :return: None
        """
        if pool_type == 'avg':
            mp = AveragePooling2D(pool_size=(3, 3),
                                  strides=(1, 1),
                                  padding='same')(input_layer)
        elif pool_type == 'max':
            mp = MaxPooling2D(pool_size=(3, 3),
                              strides=(1, 1),
                              padding='same')(input_layer)
        else:
            raise ValueError('set pool_type max for MaxPooling2D or avg for AveragePooling2D')
        setattr(self, f'inception_{addition_name}_pool_conv',
                Conv2D(filters=pool_filters,
                       kernel_size=(1, 1),
                       padding='same',

                       kernel_regularizer=l2(0.01),
                       name=f'inception_{addition_name}_pool_conv')(mp))
        setattr(self, f'inception_{addition_name}_pool_bn',
                BatchNormalization(name=f'inception_{addition_name}_pool_bn')(
                    getattr(self, f'inception_{addition_name}_pool_conv')))
        setattr(self, f'inception_{addition_name}_pool_act',
                Activation('relu', name=f'inception_{addition_name}_pool_act')(
                    getattr(self, f'inception_{addition_name}_pool_bn')))

        setattr(self, f'inception_{addition_name}_1x1_conv',
                Conv2D(filters=conv_filters,
                       kernel_size=(1, 1),
                       padding='same',
                       kernel_regularizer=l2(0.01),
                       name=f'inception_{addition_name}_1x1_conv')(input_layer))
        setattr(self, f'inception_{addition_name}_1x1_bn',
                BatchNormalization(name=f'inception_{addition_name}_1x1_bn')
                (getattr(self, f'inception_{addition_name}_1x1_conv')))
        setattr(self, f'inception_{addition_name}_1x1_act',
                Activation('relu', name=f'inception_{addition_name}_1x1_act')(
                    getattr(self, f'inception_{addition_name}_1x1_bn')))

    def create_interception_block(self, input_layer, addition_name='',
                                    conv1_5x5=None, conv2_5x5=None,
                                    conv1_3x3=None, conv2_3x3=None,
                                    conv_filters=None, pool_filters=None,
                                    pool_type='max', addition_pool=False):
        """
        Create inception block
        :param input_layer: the last layer
        :param addition_name:  index in the naming system (idx of inception layer)
        :param conv1_5x5: num of filters in conv1_5x5 layer; None if that layer doesn't need
        :param conv2_5x5: num of filters in conv2_5x5 layer; None if that layer doesn't need
        :param conv1_3x3: num of filters in conv1_3x3 layer; None if that layer doesn't need
        :param conv2_3x3: num of filters in conv2_3x3 layer; None if that layer doesn't need
        :param conv_filters: num of filters in conv2 layer; None if that layer doesn't need
        :param pool_filters: num of filters in conv1 layer; None if that layer doesn't need
        :param pool_type:  'max' - use MaxPooling or 'avg' - use AveragePooling
        :param addition_pool: add a stream with MaxPooling layer
        :return: None
        """
        layers_set_list = []
        strides = (1, 1) if not addition_pool else (2, 2)

        if conv_filters is not None and pool_filters is not None:
            self.create_conv_pool_layers_set(input_layer, addition_name=addition_name, conv_filters=conv_filters,
                                                          pool_filters=pool_filters, pool_type=pool_type)
            layers_set_list.extend([getattr(self, f'inception_{addition_name}_1x1_act'),
                                    getattr(self, f'inception_{addition_name}_pool_act')])

        if conv1_5x5 is not None and conv2_5x5 is not None:
            setattr(self, f'inception_{addition_name}_5x5_conv1',
                          Conv2D(filters=conv1_5x5,
                                 kernel_size=(1, 1),
                                 padding='same',
                                 kernel_regularizer=l2(0.01),
                                 name=f'inception_{addition_name}_5x5_conv1')(input_layer))
            setattr(self, f'inception_{addition_name}_5x5_bn1',
                    BatchNormalization(name=f'inception_{addition_name}_5x5_bn1')(
                        getattr(self, f'inception_{addition_name}_5x5_conv1')))
            setattr(self, f'inception_{addition_name}_5x5_act1',
                    Activation('relu', name=f'inception_{addition_name}_5x5_act1')(
                        getattr(self, f'inception_{addition_name}_5x5_bn1')))

            setattr(self, f'inception_{addition_name}_5x5_conv2',
                          Conv2D(filters=conv2_5x5,
                                 kernel_size=(5, 5),
                                 strides=strides,
                                 padding='same',
                                 kernel_regularizer=l2(0.01),
                                 name=f'inception_{addition_name}_5x5_conv2')
                          (getattr(self, f'inception_{addition_name}_5x5_act1')))
            setattr(self, f'inception_{addition_name}_5x5_bn2',
                    BatchNormalization(name=f'inception_{addition_name}_5x5_bn2')(
                        getattr(self, f'inception_{addition_name}_5x5_conv2')))
            setattr(self, f'inception_{addition_name}_5x5_act2',
                    Activation('relu', name=f'inception_{addition_name}_5x5_act2')(
                        getattr(self, f'inception_{addition_name}_5x5_bn2')))

            layers_set_list.append(getattr(self, f'inception_{addition_name}_5x5_act2'))

        if conv1_3x3 is not None and conv2_3x3 is not None:
            setattr(self, f'inception_{addition_name}_3x3_conv1',
                          Conv2D(filters=conv1_3x3,
                                 kernel_size=(1, 1),
                                 padding='same',
                                 kernel_regularizer=l2(0.01),
                                 name=f'inception_{addition_name}_3x3_conv1')(input_layer))
            setattr(self, f'inception_{addition_name}_3x3_bn1',
                    BatchNormalization(name=f'inception_{addition_name}_3x3_bn1')(
                        getattr(self, f'inception_{addition_name}_3x3_conv1')))
            setattr(self, f'inception_{addition_name}_3x3_act1',
                    Activation('relu', name=f'inception_{addition_name}_3x3_act1')(
                        getattr(self, f'inception_{addition_name}_3x3_bn1')))

            setattr(self, f'inception_{addition_name}_3x3_conv2',
                          Conv2D(filters=conv2_3x3,
                                 kernel_size=(3, 3),
                                 strides=strides,
                                 padding='same',

                                 kernel_regularizer=l2(0.01),
                                 name=f'inception_{addition_name}_3x3_conv2')
                          (getattr(self, f'inception_{addition_name}_3x3_act1')))
            setattr(self, f'inception_{addition_name}_3x3_bn2',
                    BatchNormalization(name=f'inception_{addition_name}_3x3_bn2')
                    (getattr(self, f'inception_{addition_name}_3x3_conv2')))
            setattr(self, f'inception_{addition_name}_3x3_act2',
                    Activation('relu', name=f'inception_{addition_name}_3x3_act2')(
                        getattr(self, f'inception_{addition_name}_3x3_bn2')))
            layers_set_list.append(getattr(self, f'inception_{addition_name}_3x3_act2'))

        if addition_pool:
            mp = MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding='same')(input_layer)
            layers_set_list.append(mp)

        out_layer = Concatenate(axis=3)(layers_set_list)
        return out_layer

    def make_prediction(self, data):
        """
        Data encodeing
        :param data: list with images data
        :return: list with encoded images - predictions
        """
        predictions = self.model.predict(data)
        return predictions

    def data_encoding(self, data):
        """
        Encoding of data
        :param data: dict {name of image: image data}
        :return: dict {name of image: encoded data}
        """
        data_for_prediction = np.array(list(data.values()))
        predictions = self.make_prediction(data_for_prediction)
        encoded_dict = {list(data.keys())[idx]: predictions[idx] for idx in range(len(predictions))}
        return encoded_dict

    def model_evaluation(self, data, threshold=0.7):
        """
        Make predictions for data and creation of dict with likelyhood vectors
        :param data: numpy array (N, H, W, C)
        :param threshold: float number with max of distance between vectors
        :return: dict {idx of image: set of names}
        """
        predictions = self.make_prediction(data)
        dist_mtx = pairwise_distances(predictions)
        mask = dist_mtx <= threshold
        result_dict = {idx: set(np.argwhere(mask[idx]).reshape(-1)) ^ {idx} for idx in range(mask.shape[0])}
        return result_dict

    def create_model_plot(self):
        """
        Create plot with model graph
        :return: None
        """
        plot_model(self.model, to_file='model.png', show_shapes=True, show_layer_names=True)


def prepare_data(path_to_folder, img_size=(96, 96, 3)):
    """
    Load images from folder
    :param path_to_folder: input path to folder
    :param img_size: 3-dim image size
    :return:  input dictionary {filename: image}, numpy array (num, h, w, c)
    """
    files = os.listdir(path_to_folder)
    load_func = lambda filename: img_to_array(load_img(
                                                        os.path.join(path_to_folder, filename),
                                                        grayscale=False,
                                                        color_mode='rgb',
                                                        target_size=img_size,
                                                        interpolation='nearest'
                                                    ))

    input_dict = {filename: load_func(filename) for filename in files}
    return input_dict


if __name__ =='__main__':
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

    parser = ArgumentParser()
    parser.add_argument('-i', '--input_data', help='path to folder with images',
                        default='./images/validate', required=False, type=str)
    parser.add_argument('-w', '--weights', help='path to folder with weights',
                        default='./weights', required=False, type=str)
    args = parser.parse_args()

    indata_dict = prepare_data(args.input_data, img_size=(96, 96, 3))
    data = np.array(list(indata_dict.values()))

    encoder = FaceEncoder()
    encoder.create_model_plot()
    encoder.load_weights(args.weights)

    # create the corresponding database as a dictionary
    res = encoder.data_encoding(indata_dict)
    print(res)

    #who is the particular person and comparing linear distance between encodings with threshold=0.7.
    res = encoder.model_evaluation(data, threshold=0.7)
    print(res)
