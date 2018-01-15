#!/usr/bin/env python
# -*- coding: utf-8 -*-

import paddle.v2 as paddle
import numpy as np
import sys
import logging

class LstmClassify(object):
    def __init__(self,
                 batch_size=200,
                 time_step=5,
                 hide_size=256,
                 input_size=2048,
                 output_size=1,
                 lstm_depth=5,
                 mix_hide_lr=6e-3,
                 lstm_lr=6e-3,
                 drop_out=0.5,
                 gradient_clipping=5,
                 act=paddle.activation.Relu(),
                 gate_act=paddle.activation.Sigmoid(),
                 state_act=paddle.activation.Sigmoid()):
        
        # Logger configuration
        self.__logger = logging.getLogger("LSTM Logger")
        self.__logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)
        sh.setFormatter(formatter)
        self.__logger.addHandler(sh)
        #Lstm params configuration
        self.__batch_size = batch_size
        self.__time_step = time_step
        self.__hide_size = hide_size
        self.__input_size = input_size
        self.__output_size = output_size
        self.__lstm_depth = lstm_depth
        self.__mix_hide_lr = mix_hide_lr
        self.__lstm_lr = lstm_lr
        self.__drop_out = drop_out
        self.__act = act
        self.__gate_act = gate_act
        self.__state_act = state_act
        self.__gradient_clipping = gradient_clipping
        self.__build_network()
        
    def __build_network(self):
        X = paddle.layer.data(name="lstm_X", 
            type=paddle.data_type.dense_vector_sequence(self.__input_size))
        Y = paddle.layer.data(name="lstm_Y", 
            type=paddle.data_type.integer_value(self.__output_size))
        
        mix_attr = paddle.attr.ExtraAttr(error_clipping_threshold=self.__gradient_clipping,
                                         drop_out=self.__drop_out)
        hide_para_attr = paddle.attr.Param(initial_std=self.__default_std,
                                           learning_rate=self.__mix_hide_lr,
                                           gradient_clipping_threshold=self.__gradient_clipping)
        lstm_para_attr = paddle.attr.Param(initial_std=self.__default_std,
                                           learning_rate=self.__lstm_lr,
                                           gradient_clipping_threshold=self.__gradient_clipping)
        
        hide_0 = paddle.layer.mixed(input=paddle.layer.full_matrix_projection(
                                        input=X, param_attr=hide_para_attr),
                                    size=self.__hide_size,
                                    act=paddle.activation.Linear(),
                                    layer_attr=mix_attr,
                                    bias_attr=None)
        
        lstm_layer_0 = paddle.layer.lstmemory(input=hide_0,
                                              reverse=False,
                                              bias_attr=None,
                                              param_attr=lstm_para_attr,
                                              act=self.__act,
                                              gate_act=self.__gate_act,
                                              state_act=self.__state_act)
                                       
        input_temp = lstm_layer_0
    
        for i in range(1, self.__lstm_depth):
            hide = paddle.layer.mixed(input=paddle.layer.full_matrix_projection(
                                        input=input_temp, param_attr=hide_para_attr),
                                      size=self.__hide_size,
                                      act=paddle.activation.Linear(),
                                      layer_attr=mix_attr,
                                      bias_attr=None)
                                  
            lstm = paddle.layer.lstmemory(input=hide,
                                          reverse=False,
                                          bias_attr=None,
                                          param_attr=lstm_para_attr,
                                          act=self.__act,
                                          gate_act=self.__gate_act,
                                          state_act=self.__state_act)
            input_temp = lstm
        
        self.__output = paddle.layer.fc(input=input_temp, 
            size=self.__output_size, act=paddle.activation.Softmax())
        cost = paddle.layer.cross_entropy_cost(input=self.__output, label=Y)
        self.__parameters = paddle.parameters.create(cost)
        optimizer = paddle.optimizer.Adam()
        self.__trainer = paddle.trainer.SGD(cost=cost, parameters=self.__parameters, 
            update_equation=optimizer)
        
    def __data_reader(self):
        def reader():
            """Read from train_x and train_y
            
            """
            for i in range(self.__train_x.shape[0]):
                yield self.__train_x[i], self.__train_y[i]
        return reader
    
    def __event_handler(self, event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % self.__log_every_n == 0:
                self.__logger.info("Training Info  epoch: {}  batch: {}  cost: {:.4f}..."
                                  .format(event.pass_id,
                                          event.batch_id,
                                          event.cost))
            self.__pass_cost += event.cost
        if isinstance(event, paddle.event.EndPass):
            #Save model
            if self.__cost_val > self.__pass_cost:
                self.__cost_val = self.__pass_cost
                with open(self.__save_path, "w") as file:
                    self.__parameters.to_tar(file)
                self.__logger.info("Save LSTM model. epoch: {}  cost: {:.4f}..."
                                  .format(event.pass_id, self.__cost_val))
            self.__pass_cost = 0.0
    
    def lstm_train(self,
                   train_set_num,
                   train_x,
                   train_y,
                   save_path,
                   log_every_n=1,
                   train_pass_num=2000,
                   cost_val=100000.0,
                   pass_cost=0.0):
        """
            Args:
            :param train_x: feature set of train data set(shape is [train_set_num, time_step, input_size]).
            :type train_x: array
            :param train_y: label set of train data set(shape is [train_set_num, output_size]).
            :type train_y: array
        """
        self.__train_x = train_x
        self.__train_y = train_y
        self.__save_path = save_path
        self.__cost_val = cost_val
        self.__pass_cost = pass_cost
        self.__log_every_n = log_every_n
        feeding = {"train_X": 0, "label_Y": 1}
        self.__trainer.train(reader=paddle.minibatch.batch(paddle.reader.shuffle(
            self.__data_reader(), buf_size=train_set_num), batch_size=self.__batch_size), 
            feeding=feeding, event_handler=self.__event_handler, num_passes=train_pass_num)
        
    def lstm_predict(self, features, load_path):
        """
            Args:
        
            :param features: feature set of predicting data set.(shape=[num_of_batch, batch_size, time_step, input_size])
            :type features: array
            :param load_path: file path to load model.
            :type load_path: string
        """
        with open(load_path, "r") as file:
            predict_params = paddle.parameters.Parameters.from_tar(file)
        predict_feeding = {"predict_X": 0}
        predict_result = paddle.infer(output_layer=self.__output, parameters=predict_params, 
            input=features, feeding=predict_feeding)
        predict_result = np.reshape(predict_result, (-1, self.__output_size))
        ret = predict_result[0:, 1]
        return ret
        

