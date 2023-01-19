"""
Provides a class for torch model evaluation.
"""

from __future__ import absolute_import

import warnings

import torch
import numpy as np

from torchfuzz.utils.common import to_numpy
import os
import pickle
import random

from torchfuzz.utils.mutation import choose_fun

from torchfuzz.metrics.neuron_coverage import NeuronCoverage


class PyTorchModel:
    """ Class for torch model evaluation.

    Provide predict, intermediate_layer_outputs and adversarial_attack
    methods for model evaluation. Set callback functions for each method
    to process the results.

    Parameters
    ----------
    model : instance of torch.nn.Module
        torch model to evaluate.

    Notes
    ----------
    All operations will be done using GPU if the environment is available
    and set properly.

    """

    def __init__(self, model, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        # GPU 한개만 사용할 경우
        # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"]= "2"

        assert isinstance(model, torch.nn.Module)
        self._model = model
        # self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._device = device
        self._model.eval()
        self._model.to(self._device)

        self._neuron_array = np.array([])   # 전체 커버리지 저장


    def run_fuzzing(self, dataloader, isTrain=True, isRandom=0, threshold=0.5, params_list=0):
        """Get the intermediate layer outputs of the model.

        The method will use the model to do prediction batch by batch. For
        every batch, the the intermediate layer outputs will be captured and
        callback functions will be invoked. all intermediate layer output
        will be passed to the callback functions to do further process.

        Parameters
        ----------
        dataset : instance of torch.utils.data.Dataset
            Dataset from which to load the data.
        isTrain: boolean
            Check wheter dataset is train data or test data
        isRandom: interger
            0 when want to check all parameters else positive integer
        threshold: float
            Neuron coverage activate threshold
        params_list: two-dimensional list or empty
            Empty if want to use base parameters else two-dimensional list of parameters
        
        See Also
        --------
        :class:`metrics.neuron_coverage.NeuronCoverage`

        """
        batch_size = 1

        if os.path.isfile('./cache/nc_arr.npy'):
            self._neuron_array = np.load('./cache/nc_arr.npy')
            if isTrain:
                return 0

        self._neuron_coverage = NeuronCoverage(threshold=0.5)
        callbacks = [self._neuron_coverage.update, self._neuron_coverage.report]

        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        y_mini_batch_outputs = []
        hook_handles = []
        intermediate_layers = self._intermediate_layers(self._model)

        def hook(module, input, output):
                y_mini_batch_outputs.append(output)

        for intermediate_layer in intermediate_layers:
            handle = intermediate_layer.register_forward_hook(hook)
            hook_handles.append(handle)

        params_list = self.get_params_list(params_list=params_list)

        corpus_mut_img_list = list()
        corpus_label_list = list()
        corpus_ori_img_list = list()
        corpus_mut_par_list = list()

        crash_mut_img_increase_list = list()
        crash_label_increase_list = list()
        crash_ori_img_increase_list = list()
        crash_mut_par_increase_list = list()

        crash_mut_img_no_increase_list = list()
        crash_label_no_increase_list = list()
        crash_ori_img_no_increase_list = list()
        crash_mut_par_no_increase_list = list()

        with torch.no_grad():
            for data_num, data in enumerate(dataloader):
                if isinstance(data, list):
                    label = data[1].item()
                    data = data[0]

                # train data일 때
                if isTrain:
                    y_mini_batch_outputs.clear()
                    data = data.to(self._device)
                    result = self._model(data)

                    for func_num, callback in enumerate(callbacks):
                        if func_num == 0:    # 업데이트 함수
                            callback(y_mini_batch_outputs, 0)
                        else:           # 리포트 함수
                            np_neurons_updated = np.array(callback(y_mini_batch_outputs, 0)).astype(np.bool)
                            if data_num == 0:   # 첫 데이터일 때 활성 뉴런 저장
                                self._neuron_array = np.array(np_neurons_updated)

                                print('Current Total Coverage: %.6f(%d/%d)'%(self.get(isTrain=True), len([v for v in self._neuron_array if v]), len(self._neuron_array)))
                                print('================================================================================================================')

                            if not (self._neuron_array == np.array(np.logical_or(self._neuron_array, np_neurons_updated))).all(): # 새로운 커버리지 발견되었을 때
                                self._neuron_array = np.array(np.logical_or(self._neuron_array, np_neurons_updated))

                                print('Current Total Coverage: %.6f(%d/%d)'%(self.get(isTrain=True), len([v for v in self._neuron_array if v]), len(self._neuron_array)))
                                print('================================================================================================================')

                                # 커버리지 증가시 train data 커버리지 npy 저장
                                if not os.path.isdir('./cache'):
                                    os.mkdir('./cache')
                                np.save('./cache/nc_arr.npy', self._neuron_array)
                 
                # test data 일 때
                else:
                    # 이미지 tensor -> numpy로 변환하여 mutation 진행 후 numpy -> tensor
                    mut_data_list = list()
                    np_data = to_numpy(data)
                    np_data = np_data[0]
                    np_data = np.transpose(np_data, (1, 2, 0))

                    mut_param_list = list()

                    for i in range(len(params_list)):
                        for j in params_list[i]:
                            np_mut_img = choose_fun(np_data, i, j)
                            np_mut_img = np.transpose(np_mut_img, (2, 0, 1))
                            mut_data_list.append(torch.Tensor(np.expand_dims(np_mut_img, axis=0)))
                            mut_param_list.append([i, j])


                    # 파라미터 랜덤 선택 리스트
                    if isRandom > 0:
                        random_list = random.sample(range(len(mut_data_list)), isRandom)


                    ''''''''''''''''''
                    for mut_num, mut_data in enumerate(mut_data_list):
                        if isRandom > 0 and mut_num not in random_list:   # 랜덤 선택이고 선택 안된 파라미터이면
                            continue

                        y_mini_batch_outputs.clear()
                        mut_data = mut_data.to(self._device)
                        result = self._model(mut_data)
                        result = result.cpu().numpy()
                        while 'ndarray' in str(type(result[0])):
                            result = result[0]
                        
                        result_num = result.argmax()    # 모델 예측 값
                        isCorpus = result_num == label

                        if isCorpus:
                            mut_img, label, ori_img, mut_par = self.save_img(mut_img=mut_data, label=label, ori_img=np_data, mut_par=mut_param_list[mut_num])
                            corpus_mut_img_list.append(mut_img)
                            corpus_label_list.append(label)
                            corpus_ori_img_list.append(ori_img)
                            corpus_mut_par_list.append(mut_par)

                        for func_num, callback in enumerate(callbacks):
                            if func_num == 0:    # 업데이트 함수
                                callback(y_mini_batch_outputs, 0)
                            else:           # 리포트 함수
                                np_neurons_updated = np.array(callback(y_mini_batch_outputs, 0)).astype(np.bool)
                                if not (self._neuron_array == np.array(np.logical_or(self._neuron_array, np_neurons_updated))).all(): # 새로운 커버리지 발견되었을 때
                                    updated_neurons_test_arr = np.logical_xor(np.logical_or(self._neuron_array, np_neurons_updated), self._neuron_array)
                                    print('This image has new neuron: %.6f(%d/%d)'%(self.get(isTrain=False, neuron_array=updated_neurons_test_arr), len([v for v in updated_neurons_test_arr if v]), len(updated_neurons_test_arr)))
                                    print('===============================================================================================================')

                                    if not isCorpus:    # crash, coreverage 증가 샘플 저장
                                        mut_img, label, ori_img, mut_par = self.save_img(mut_img=mut_data, label=label, ori_img=np_data, mut_par=mut_param_list[mut_num])
                                        crash_mut_img_increase_list.append(mut_img)
                                        crash_label_increase_list.append(label)
                                        crash_ori_img_increase_list.append(ori_img)
                                        crash_mut_par_increase_list.append(mut_par)

                                else: # 새로운 커버리지 발견 x
                                    if not isCorpus:    # crash, coverage 증가 x 샘플 저장
                                        mut_img, label, ori_img, mut_par = self.save_img(mut_img=mut_data, label=label, ori_img=np_data, mut_par=mut_param_list[mut_num])
                                        crash_mut_img_no_increase_list.append(mut_img)
                                        crash_label_no_increase_list.append(label)
                                        crash_ori_img_no_increase_list.append(ori_img)
                                        crash_mut_par_no_increase_list.append(mut_par)

                    if not os.path.isdir('./cache'):
                        os.mkdir('./cache')
                    with open('./cache/corpus.pickle', 'wb') as f:
                        pickle.dump([corpus_mut_img_list, corpus_label_list, corpus_ori_img_list, corpus_mut_par_list], f, pickle.HIGHEST_PROTOCOL)
                    with open('./cache/crash_increase.pickle', 'wb') as f:
                        pickle.dump([crash_mut_img_increase_list, crash_label_increase_list, crash_ori_img_increase_list, crash_mut_par_increase_list], f, pickle.HIGHEST_PROTOCOL)
                    with open('./cache/crash_no_increase.pickle', 'wb') as f:
                        pickle.dump([crash_mut_img_no_increase_list, crash_label_no_increase_list, crash_ori_img_no_increase_list, crash_mut_par_no_increase_list], f, pickle.HIGHEST_PROTOCOL)
                    with open("./cache/Readme.txt", "w") as f:
                        f.write('''corpus.pickle: 모델이 정답으로 분류한 데이터\ncrash_increase.pickle: 모델이 오답으로 분류, 커버리지 증가 데이터\ncrash_no_increase.pickle: 모델이 오답으로 분류, 커버리지 증가 x 데이터\n-------------------------------------------------------------\n파일 구조\n[퍼징 이미지 리스트, 라벨 리스트, 원본 이미지 리스트, 뮤테이션 파라미터 리스트]''')

        for handle in hook_handles:
            handle.remove()


    def _intermediate_layers(self, module):
        """Get the intermediate layers of the model.

        The method will get some intermediate layers of the model which might
        be useful for neuron coverage computation. Some layers such as dropout
        layers are excluded empirically.

        Returns
        -------
        list of torch.nn.modules
            Intermediate layers of the model.

        """
        intermediate_layers = []
        for submodule in module.children():
            if len(submodule._modules) > 0:
                intermediate_layers += self._intermediate_layers(submodule)
            else:
                if 'Dropout' in str(submodule.type):
                    continue
                intermediate_layers.append(submodule)
        return intermediate_layers
    

    def get(self, isTrain=False, neuron_array=[]):
        """Get model neuron coverage.
        Parameters
        ----------
        isTrain : boolean
            Is train stage or test stage
        np_neurons_updated : np array
            Activated neuron array
        Returns
        -------
        float
            Model neuron coverage with parameter as the neuron activation threshold.
        Notes
        -------
        The parameter threshold must be one value in the list thresholds.
        """
        if isTrain:
            return len([v for v in self._neuron_array if v]) / len(self._neuron_array) if len(self._neuron_array) != 0 else 0
        else:
            return len([v for v in neuron_array if v]) / len(neuron_array) if len(neuron_array) != 0 else 0


    def save_img(self, mut_img, label, ori_img, mut_par):
        mut_img = mut_img[0]
        mut_img = to_numpy(mut_img)
        mut_img = np.transpose(mut_img, (1, 2, 0))
        
        return mut_img, label, ori_img, mut_par

    def get_params_list(self, params_list=0):
        if 'list' not in str(type(params_list)):
            # params_list = [  # 테스트 파라미터
            #     [-20, -10, -5, 5, 10, 20],  # translation       0
            #     [5, 7, 12, 13, 15, 17],  # scale             1
            #     [-6, -5, -3, 3, 5, 6],  # shear             2
            #     [1, 2, 13, 20],  # contrast                 3
            #     [-60, -50, -40, 40, 50, 60],  # rotation    4
            #     [-90, -80, -70, 70, 80, 90],  # brightness  5
            #     [1, 2, 3, 5, 7, 9],  # blur                 6
            #     [1, 3, 5, 7, 9, 11],  # GaussianBlur        7
            #     [1, 3, 5],  # MedianBlur                    8
            #     [6, 9]  # bilateraFilter                    9
            #     # [1, 100],       # pixel_change
            #     # [0, 150],       # noise1
            #     # [0, 0.05]       # noise2
            # ]
            params_list = [  # 테스트 파라미터
                [-3, -2, -1, 1, 2, 3],  # translation       0
                [7, 8, 10, 11, 12],  # scale             1
                [-6, -5, -3, 3, 5, 6],  # shear             2
                [5, 7, 9, 11, 13],  # contrast                 3
                [-50, -40, -30, 30, 40, 50],  # rotation    4
                [-20, -10, 10, 20],  # brightness  5
                [1, 2, 3, 5, 7, 9],  # blur                 6
                [1, 3, 5, 7, 9, 11],  # GaussianBlur        7
                [1, 3, 5],  # MedianBlur                    8
                [6, 9]  # bilateraFilter                    9
                # [1, 100],       # pixel_change
                # [0, 150],       # noise1
                # [0, 0.05]       # noise2
            ]
        return params_list