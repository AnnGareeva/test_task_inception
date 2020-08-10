import os
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from argparse import ArgumentParser

from tensorflow.keras.layers import Input, Dense, \
                                    Conv2D, MaxPooling2D, \
                                    AveragePooling2D, Dropout, \
                                    BatchNormalization, Flatten, \
                                    Concatenate, GlobalAveragePooling2D, \
                                    Lambda, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class FaceEncoder():
    def __init__(self):
        self.order_w_name_idx = (1, 2, 3, 0)
        self.order_b_name_idx = (1, 0)

        input_size = Input(shape=(3, 96, 96))

        conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', name='conv1', data_format='channels_first')(input_size)
        bn1 = BatchNormalization(name='bn1', axis=1)(conv1)
        act = Activation('relu')(bn1)

        mp = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_first')(act)

        conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', name='conv2', data_format='channels_first')(mp)
        bn2 = BatchNormalization(name='bn2', axis=1)(conv2)
        act = Activation('relu')(bn2)

        conv3 = Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3', data_format='channels_first')(act)
        bn3 = BatchNormalization(name='bn3', axis=1)(conv3)
        act = Activation('relu')(bn3)

        mp = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_first')(act)

        # Inception blocks
        inception_3a = self.create_full_inception_layer(mp, addition_name='3a',
                                                           conv1_5x5=16, conv2_5x5=32,
                                                           conv1_3x3=96, conv2_3x3=128,
                                                           conv_filters=64, pool_filters=32,
                                                           pool_type='max')

        inception_3b = self.create_full_inception_layer(inception_3a, addition_name='3b',
                                                           conv1_5x5=32, conv2_5x5=64,
                                                           conv1_3x3=96, conv2_3x3=128,
                                                           conv_filters=64, pool_filters=64,
                                                           pool_type='avg')

        inception_3c = self.create_connection_inception_layer(inception_3b, addition_name='3c',
                                                           conv1_5x5=32, conv2_5x5=64,
                                                           conv1_3x3=128, conv2_3x3=256)

        inception_4a = self.create_full_inception_layer(inception_3c, addition_name='4a',
                                                           conv1_5x5=32, conv2_5x5=64,
                                                           conv1_3x3=96, conv2_3x3=192,
                                                           conv_filters=256, pool_filters=128,
                                                           pool_type='avg')

        inception_4e = self.create_connection_inception_layer(inception_4a, addition_name='4e',
                                                           conv1_5x5=64, conv2_5x5=128,
                                                           conv1_3x3=160, conv2_3x3=256)

        inception_5a = self.create_5_inception(inception_4e, addition_name='5a',
                                                           conv1_3x3=96, conv2_3x3=384,
                                                           conv_filters=256, pool_filters=96,
                                                           pool_type='avg')
        inception_5b = self.create_5_inception(inception_5a, addition_name='5b',
                                                           conv1_3x3=96, conv2_3x3=384,
                                                           conv_filters=256, pool_filters=96,
                                                           pool_type='max')

        # Common last layers
        avg = AveragePooling2D(name='avg_pool', pool_size=(3, 3), strides=(1, 1),
                               data_format='channels_first',  padding='valid')(inception_5b)
        flatten_layer = Flatten()(avg)
        dense_layer = Dense(units=128)(flatten_layer)

        model = Model(input_size, dense_layer)
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
        # TODO: regular expression
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

    def create_5x5(self, input_layer, addition_name, conv1_5x5, conv2_5x5, strides=(1, 1)):
        """
        Create 5x5 sequences for inception
        :param input_layer: the last layer
        :param addition_name: index in the naming system (idx of inception layer)
        :param conv1_5x5: num of filters in conv1_5x5 layer
        :param conv2_5x5: num of filters in conv2_5x5 layer
        :param strides: tuple with strides
        :return: out layer
        """
        conv15x5 = Conv2D(filters=conv1_5x5,
                          kernel_size=(1, 1),
                          padding='same',
                          data_format='channels_first',
                          name=f'inception_{addition_name}_5x5_conv1')(input_layer)
        bn15x5 = BatchNormalization(name=f'inception_{addition_name}_5x5_bn1', axis=1)(conv15x5)
        act15x5 = Activation('relu', name=f'inception_{addition_name}_5x5_act1')(bn15x5)
        conv25x5 = Conv2D(filters=conv2_5x5,
                          kernel_size=(5, 5),
                          strides=strides,
                          padding='same',
                          data_format='channels_first',
                          name=f'inception_{addition_name}_5x5_conv2')(act15x5)
        bn25x5 = BatchNormalization(name=f'inception_{addition_name}_5x5_bn2', axis=1)(conv25x5)
        act25x5 = Activation('relu', name=f'inception_{addition_name}_5x5_act2')(bn25x5)
        return act25x5

    def create_3x3(self, input_layer, addition_name, conv1_3x3, conv2_3x3, strides=(1, 1)):
        """
        Create 3x3 sequences for inception
        :param input_layer: the last layer
        :param addition_name: index in the naming system (idx of inception layer)
        :param conv1_3x3: num of filters in conv1_3x3 layer
        :param conv2_3x3: num of filters in conv2_3x3 layer
        :param strides: tuple with strides
        :return: out layer
        """
        conv13x3 = Conv2D(filters=conv1_3x3,
                          kernel_size=(1, 1),
                          padding='same',
                          data_format='channels_first',
                          name=f'inception_{addition_name}_3x3_conv1')(input_layer)
        bn13x3 = BatchNormalization(name=f'inception_{addition_name}_3x3_bn1', axis=1)(conv13x3)
        act13x3 = Activation('relu', name=f'inception_{addition_name}_3x3_act1')(bn13x3)
        conv23x3 = Conv2D(filters=conv2_3x3,
                          kernel_size=(3, 3),
                          strides=strides,
                          padding='same',
                          data_format='channels_first',
                          name=f'inception_{addition_name}_3x3_conv2')(act13x3)
        bn23x3 = BatchNormalization(name=f'inception_{addition_name}_3x3_bn2', axis=1)(conv23x3)
        act23x3 = Activation('relu', name=f'inception_{addition_name}_3x3_act2')(bn23x3)
        return act23x3

    def create_1x1(self,  input_layer, addition_name, conv_filters, pool_filters, pool_type):
        """
        Create 2 additional streams of layers:
                                            input_layer -> pooling-> conv_1
                                            input_layer -> conv_2
        :param input_layer: the last layer
        :param addition_name: index in the naming system (idx of inception layer)
        :param conv_filters: num of filters in conv_2 layer
        :param pool_filters: num of filters in conv_1 layer
        :param pool_type: 'max' - use MaxPooling or 'avg' - use AveragePooling
        :return: out 2 layers: pool_conv, and conv1x1
        """
        if pool_type == 'avg':
            mp = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first', padding='same')(input_layer)
        elif pool_type == 'max':
            mp = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_first', padding='same')(input_layer)
        else:
            raise ValueError('set pool_type max for MaxPooling2D or avg for AveragePooling2D')
        pool_conv = Conv2D(filters=pool_filters,
                           kernel_size=(1, 1),
                           padding='same',
                           data_format='channels_first',
                           name=f'inception_{addition_name}_pool_conv')(mp)
        pool_bn = BatchNormalization(name=f'inception_{addition_name}_pool_bn', axis=1)(pool_conv)
        pool_act = Activation('relu', name=f'inception_{addition_name}_pool_act')(pool_bn)

        conv1x1 = Conv2D(filters=conv_filters,
                         kernel_size=(1, 1),
                         padding='same',
                         data_format='channels_first',
                         name=f'inception_{addition_name}_1x1_conv')(input_layer)
        bn1x1 = BatchNormalization(name=f'inception_{addition_name}_1x1_bn', axis=1)(conv1x1)
        act1x1 = Activation('relu', name=f'inception_{addition_name}_1x1_act')(bn1x1)

        return pool_act, act1x1

    def create_full_inception_layer(self, input_layer, addition_name,
                                    conv1_5x5, conv2_5x5,
                                    conv1_3x3, conv2_3x3,
                                    conv_filters, pool_filters,
                                    pool_type):
        """
        Create inception block with conv5x5, conv3x3, conv1x1
        :param input_layer: the last layer
        :param addition_name: index in the naming system (idx of inception layer)
        :param conv1_5x5: num of filters in conv1_5x5 layer
        :param conv2_5x5: num of filters in conv2_5x5 layer
        :param conv1_3x3: num of filters in conv1_3x3 layer
        :param conv2_3x3: num of filters in conv2_3x3 layer
        :param conv_filters: num of filters in conv_2 layer
        :param pool_filters: num of filters in conv_1 layer
        :param pool_type: 'max' - use MaxPooling or 'avg' - use AveragePooling
        :return: out layer
        """
        strides = (1, 1)
        pool_act, act1x1 = self.create_1x1(input_layer, addition_name, conv_filters, pool_filters, pool_type)
        act25x5 = self.create_5x5(input_layer, addition_name, conv1_5x5, conv2_5x5, strides)
        act23x3 = self.create_3x3(input_layer, addition_name, conv1_3x3, conv2_3x3, strides)
        out_layer = Concatenate(axis=1)([pool_act, act1x1, act25x5, act23x3])
        return out_layer

    def create_connection_inception_layer(self, input_layer, addition_name,
                                                conv1_5x5, conv2_5x5,
                                                conv1_3x3, conv2_3x3):
        """
        Create inception block with conv5x5, conv3x3
        :param input_layer: the last layer
        :param addition_name: index in the naming system (idx of inception layer)
        :param conv1_5x5: num of filters in conv1_5x5 layer
        :param conv2_5x5: num of filters in conv2_5x5 layer
        :param conv1_3x3: num of filters in conv1_3x3 layer
        :param conv2_3x3: num of filters in conv2_3x3 layer
        :return: out layer
        """
        strides = (2, 2)
        act25x5 = self.create_5x5(input_layer, addition_name, conv1_5x5, conv2_5x5, strides)
        act23x3 = self.create_3x3(input_layer, addition_name, conv1_3x3, conv2_3x3, strides)
        mp = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_first')(input_layer)
        out_layer = Concatenate(axis=1)([act25x5, act23x3, mp])
        return out_layer

    def create_5_inception(self, input_layer, addition_name,
                                   conv1_3x3, conv2_3x3,
                                   conv_filters, pool_filters,
                                   pool_type):
        """
        Create inception block for inception (5 part)
        :param input_layer: the last layer
        :param addition_name: index in the naming system (idx of inception layer)
        :param conv1_3x3: num of filters in conv1_3x3 layer
        :param conv2_3x3: num of filters in conv2_3x3 layer
        :param conv_filters: num of filters in conv_2 layer
        :param pool_filters: num of filters in conv_1 layer
        :param pool_type: 'max' - use MaxPooling or 'avg' - use AveragePooling
        :return: out layer
        """
        strides = (1, 1)
        act23x3 = self.create_3x3(input_layer, addition_name, conv1_3x3, conv2_3x3, strides)
        pool_act, act1x1 = self.create_1x1(input_layer, addition_name, conv_filters, pool_filters, pool_type)
        out_layer = Concatenate(axis=1)([pool_act, act1x1, act23x3])
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

    def model_evaluation(self, indata_dict, threshold=0.7):
        """
        Make predictions for data and creation of dict with likelyhood vectors
        :param indata_dict: dict {name of image: image data}
        :param threshold: float number with max of distance between vectors
        :return: dict {idx of image: set of names}
        """
        data = np.array(list(indata_dict.values())) #numpy array (N, C, H, W)
        predictions = self.make_prediction(data)
        dist_mtx = pairwise_distances(predictions)
        print(dist_mtx)
        mask = dist_mtx <= threshold
        result_dict = {idx: set(np.argwhere(mask[idx]).reshape(-1)) ^ {idx} for idx in range(mask.shape[0])}
        return result_dict


def prepare_data(path_to_folder, img_size=(96, 96, 3)):
    """
    Load images from folder
    :param path_to_folder: input path to folder
    :param img_size: 3-dim image size
    :return:  input dictionary {filename: image}, numpy array (num, h, w, c)
    """
    files = os.listdir(path_to_folder)
    load_func = lambda filename: np.einsum('hwc->chw', img_to_array(load_img(os.path.join(path_to_folder, filename),
                                                                                grayscale=False,
                                                                                color_mode='rgb',
                                                                                target_size=img_size,
                                                                                interpolation='nearest')))

    input_dict = {filename: load_func(filename) for filename in files}
    return input_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_data', help='path to folder with images',
                        default='./images/validate', required=False, type=str)
    parser.add_argument('-w', '--weights', help='path to folder with weights',
                        default='./weights', required=False, type=str)
    args = parser.parse_args()

    encoder = FaceEncoder()
    encoder.load_weights(args.weights)

    indata_dict = prepare_data(args.input_data, img_size=(96, 96, 3))
    res = encoder.data_encoding(indata_dict)
    print(res)

    res = encoder.model_evaluation(indata_dict, threshold=0.7)
    print(res)
