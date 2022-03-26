# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 16:33
# @Author  : dongchao yang
# @File    : train.py
import os
import datetime

import uuid
import glob
from pathlib import Path
import fire

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from ignite.contrib.handlers import ProgressBar, CustomPeriodicEvent, param_scheduler
from ignite.engine import (Engine, Events)
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, RunningAverage, Precision, Recall
from ignite.utils import convert_tensor
from tabulate import tabulate
import random
import dataset
import models
import utils
import metrics
import losses
import torch.backends.cudnn as cudnn
from sklearn.metrics import average_precision_score
from psds_eval import (PSDSEval, plot_psd_roc, plot_per_class_psd_roc)
from psds_score import get_eval_score
import logging
import torch.distributed as dist

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(DEVICE)
seed = 2021
if seed is not None:
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True

class Runner(object):
    """Main class to run experiments with e.g., train and evaluate"""
    def __init__(self, seed=42):
        """__init__
        :param config: YAML config file
        :param **kwargs: Overwrite of yaml config
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _forward(model, batch):
        inputs, frame_level_targets, time, embedding, filenames,events = batch
        # print(events)
        # print(frame_level_time[:,0])
        # assert 1==2
        inputs = convert_tensor(inputs, device=DEVICE, non_blocking=True)
        frame_level_targets = convert_tensor(frame_level_targets.float(), device=DEVICE, non_blocking=True)
        embedding = convert_tensor(embedding, device=DEVICE, non_blocking=True)
        sim_res, ans_label = model(inputs,embedding,frame_level_targets)
        return sim_res, ans_label
    
    @staticmethod
    def _negative_loss(engine):
        return -engine.state.metrics['Loss']

    def train(self, config, **kwargs):
        """Trains a given model specified in the config file or passed as the --model parameter.
        All options in the config file can be overwritten as needed by passing --PARAM
        Options with variable lengths ( e.g., kwargs can be passed by --PARAM '{"PARAM1":VAR1, "PARAM2":VAR2}'
        :param config: yaml config file
        :param **kwargs: parameters to overwrite yaml config
        """
        config_parameters = utils.parse_config_or_kwargs(config, **kwargs) # get parameters dict according to yaml file
        outputdir = os.path.join(config_parameters['outputpath'], config_parameters['model'],
            "{}_{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'), uuid.uuid1().hex)) #  according time and uuid, we can get one file path,all of our experiment results will store in it
        # Create base dir
        Path(outputdir).mkdir(exist_ok=True, parents=True) # make dir
        logger = utils.getfile_outlogger(os.path.join(outputdir, 'train.log')) # record train process. logger obeject can help us record our process
        logger.info("Storing files in {}".format(outputdir))
        # utils.pprint_dict
        utils.pprint_dict(config_parameters, logger.info) # print yaml file content
        logger.info("Running on device {}".format(DEVICE))
        transform = utils.parse_transforms(config_parameters['transforms']) # three data augment methods
        torch.save(config_parameters, os.path.join(outputdir,'run_config.pth')) # save config_parameters
        logger.info("Transforms:")
        utils.pprint_dict(transform, logger.info, formatter='pretty') # print the details of transform
        # For Unbalanced Audioset, this is true
        sampling_kwargs = {"shuffle": True}
        logger.info("Using Sampler {}".format(sampling_kwargs))
        trainloader = dataset.getdataloader_s_test(
            config_parameters['train_data'], # feature path
            config_parameters['spk_emb_file_path'],
            transform=transform,
            batch_size=config_parameters['batch_size'],
            num_workers=config_parameters['num_workers'],**sampling_kwargs)

        cvdataloader = dataset.getdataloader_s_test(
            config_parameters['cv_data'],
            config_parameters['spk_emb_file_path'],
            transform=None,
            shuffle=False,
            batch_size=config_parameters['batch_size'],
            num_workers=config_parameters['num_workers'])
        model = getattr(models, config_parameters['model'],
                        'CRNN')(config_parameters,inputdim=64, outputdim=2,
                                **config_parameters['model_args'])
        if 'pretrained' in config_parameters and config_parameters['pretrained'] is not None:
            models.load_pretrained(model, config_parameters['pretrained'], outputdim=2)
            logger.info("Loading pretrained model {}".format(config_parameters['pretrained']))

        model = model.to(DEVICE)
        if config_parameters['optimizer'] == 'AdaBound':
            try:
                import adabound
                optimizer = adabound.AdaBound(model.parameters(), **config_parameters['optimizer_args'])
            except ImportError:
                config_parameters['optimizer'] = 'Adam'
                config_parameters['optimizer_args'] = {}
        else:
            #optimizer = getattr(torch.optim,config_parameters['optimizer'],)(model.parameters(), **config_parameters['optimizer_args']) # 加载 optimizer
            optimizer = getattr(torch.optim,config_parameters['optimizer'],)(filter(lambda p: p.requires_grad, model.parameters()), **config_parameters['optimizer_args']) # 加载 optimizer

        utils.pprint_dict(optimizer, logger.info, formatter='pretty')
        utils.pprint_dict(model, logger.info, formatter='pretty')
        if DEVICE.type != 'cpu' and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model) # 同时使用多个GPU
        criterion = getattr(losses, config_parameters['loss'])().to(DEVICE)
        def _train_batch(_, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                sim_res, ans_label = self._forward(model, batch)  # output is tuple (clip, frame, target)
                # print('decision ',decision.shape)
                # print('frame_level_target ',frame_level_target.shape)
                # assert 1==2
                loss = criterion(sim_res, ans_label)
                loss.backward()
                # Single loss
                optimizer.step()
                return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                return self._forward(model, batch)

        def thresholded_output_transform(output):
            sim_res, ans_label = output
            #y_pred = torch.round(decision) # 将输入input张量每个元素舍入到最近的整数
            y_pred  = sim_res > 0.5
            y_pred = y_pred.long()
            return y_pred, ans_label

        precision = Precision(thresholded_output_transform, average=False)
        recall = Recall(thresholded_output_transform, average=False)
        f1_score = (precision * recall * 2 / (precision + recall)).mean()
        metrics = {
            'Loss': losses.Loss_simnet(criterion),  #reimplementation of Loss, supports 3 way loss 
            'Precision': Precision(thresholded_output_transform),
            'Recall': Recall(thresholded_output_transform),
            'Accuracy': Accuracy(thresholded_output_transform),
            'F1': f1_score}
        train_engine = Engine(_train_batch)
        inference_engine = Engine(_inference)
        for name, metric in metrics.items():
            metric.attach(inference_engine, name)

        def compute_metrics(engine):
            inference_engine.run(cvdataloader) # run validate set
            results = inference_engine.state.metrics # 
            output_str_list = ["Validation Results - Epoch : {:<5}".format(engine.state.epoch)]
            for metric in metrics:
                output_str_list.append("{} {:<5.2f}".format(metric, results[metric])) # get all metric obout this validation
            logger.info(" ".join(output_str_list))
            
        pbar = ProgressBar(persist=False)
        pbar.attach(train_engine)
        if 'itercv' in config_parameters and config_parameters['itercv'] is not None:
            train_engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=config_parameters['itercv']), compute_metrics)
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, compute_metrics) # add validate process on train engine
        # Default scheduler is using patience=3, factor=0.1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config_parameters['scheduler_args']) # using scheduler with learning rate

        @inference_engine.on(Events.EPOCH_COMPLETED)
        def update_reduce_on_plateau(engine):
            logger.info(f"Scheduling epoch {engine.state.epoch}")
            val_loss = engine.state.metrics['Loss']
            if 'ReduceLROnPlateau' == scheduler.__class__.__name__:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        early_stop_handler = EarlyStopping(
            patience=config_parameters['early_stop'],
            score_function=self._negative_loss, trainer=train_engine)

        inference_engine.add_event_handler(Events.EPOCH_COMPLETED, early_stop_handler) # add early stop to inference engine
        if config_parameters['save'] == 'everyepoch':
            checkpoint_handler = ModelCheckpoint(outputdir, 'run', n_saved=5, require_empty=False)
            train_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model})
            train_engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=config_parameters['itercv']),
                checkpoint_handler, {'model': model})
        else:
            checkpoint_handler = ModelCheckpoint(
                outputdir,
                'run',
                n_saved=1,
                require_empty=False,
                score_function=self._negative_loss,
                global_step_transform=global_step_from_engine(train_engine), score_name='loss')
            inference_engine.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model})
        train_engine.run(trainloader, max_epochs=config_parameters['epochs'])
        return outputdir
    
    def evaluate(
            self,
            experiment_path: str,
            pred_file='hard_predictions_{}.txt',
            tag_file='tagging_predictions_{}.txt',
            event_file='event_{}.txt',
            segment_file='segment_{}.txt',
            class_result_file='class_result_{}.txt',
            time_ratio=10. / 1000,
            postprocessing='median',
            threshold=None,
            window_size=None,
            save_seq=True,
            sed_eval=True,  # Do evaluation on sound event detection ( time stamps, segemtn/evaluation based)
            psds_eval_ = False,
            **kwargs):
        """evaluate
        :param experiment_path: Path to already trained model using train
        :type experiment_path: str
        :param pred_file: Prediction output file, put into experiment dir
        :param time_resolution: Resolution in time (1. represents the model resolution)
        :param **kwargs: Overwrite standard args, please pass `data` and `label`
        """
        # write log file
        log = logging.getLogger('NN-CLW')
        log.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(experiment_path, 'NN-CLW.log'))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        log.addHandler(fh)
        log.addHandler(ch)

        config = torch.load(list(Path(f'{experiment_path}').glob("run_config*"))[0], map_location='cpu')
        # Use previous config, but update data such as kwargs
        config_parameters = dict(config, **kwargs)
        #config_parameters['label'] = '/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/strong_eval_choose_psds.tsv'
        # postprocessing='median'
        # Default columns to search for in data
        if 'thres' not in config_parameters.keys():
            config_parameters['thres'] = 0.5
        model_parameters = torch.load(glob.glob("{}/run_model*".format(experiment_path))[0],
                                     map_location=lambda storage, loc: storage) # load parameter    
        strong_labels_df = pd.read_csv(config_parameters['label'], sep='\t') # get 
        if not np.issubdtype(strong_labels_df['filename'].dtype, np.number):
            strong_labels_df['filename'] = strong_labels_df['filename'].apply(os.path.basename)
        if 'audiofilepath' in strong_labels_df.columns:  # In case of ave dataset, the audiofilepath column is the main column
            strong_labels_df['audiofilepath'] = strong_labels_df['audiofilepath'].apply(os.path.basename)
            colname = 'audiofilepath'  # AVE
        else:
            colname = 'filename'  # Dcase etc.
        if "event_labels" in strong_labels_df.columns:
            assert False, "Data with the column event_labels are used to train not to evaluate"
        dataloader = dataset.getdataloader_s_test(
            config_parameters['test_data'],
            config_parameters['spk_emb_file_path'],
            batch_size=8, shuffle=False)
        model = getattr(models, config_parameters['model'])(config_parameters,
            inputdim=64, outputdim=2, **config_parameters['model_args'])
        model.load_state_dict(model_parameters)
        model = model.to(DEVICE).eval()
        time_predictions, clip_predictions = [], []
        psds_time_predictions = []
        for i, th in enumerate(np.arange(0.01, 1.01, 0.01)):
            psds_time_predictions.append([])
        sequences_to_save = []
        mAP_pred, mAP_tar = [], []
        frame_num = 0
        total_num = 0
        total_pos = 0
        tongji = 0
        tongji_t = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, unit='file', leave=False): # dataloard 加载了弱标签
                inputs, frame_level_targets,time, embedding, filenames,events = batch
                sim_res, ans_label = self._forward(model, batch) # 
                print(sim_res, ans_label)
                # mAP_tar.append(frame_level_target.detach().cpu().numpy().squeeze(0))
                # mAP_pred.append(decision.detach().cpu().numpy().squeeze(0))
                y_pred  = sim_res > 0.5
                y_pred = y_pred.long()
                # print('y_pred ',y_pred.shape)
                # print('ans_label ',ans_label.shape)
                assert 1==2
                total_num += y_pred.shape[0]
                total_pos += (y_pred==ans_label).sum().detach().cpu().numpy()
                tongji += ans_label.sum().detach().cpu().numpy()
                tongji_t += ans_label.shape[0]
        print('acc ',1.0*total_pos/total_num)
        print('tongji ',tongji)
        print('tongji_t ',tongji_t)

    def train_evaluate(self, config, **kwargs):
        experiment_path = self.train(config, **kwargs) # 先进行训练
        from h5py import File
        # Get the output time-ratio factor from the model
        model_parameters = torch.load(
            glob.glob("{}/run_model*".format(experiment_path))[0],
            map_location=lambda storage, loc: storage)
        config_param = torch.load(glob.glob(
            "{}/run_config*".format(experiment_path))[0],
                                  map_location=lambda storage, loc: storage)
        # Dummy to calculate the pooling factor a bit dynamic
        # model = getattr(models, config_param['model'])(config_param,inputdim=64,
        #                 outputdim=2, **config_param['model_args'])
        # model.load_state_dict(model_parameters)
        # model.to(DEVICE)
        # time_ratio = 10.0/500
        # config_parameters = utils.parse_config_or_kwargs(config, **kwargs)
        # threshold = config_parameters.get('threshold', None)
        # postprocessing = config_parameters.get('postprocessing', 'double')
        # window_size = config_parameters.get('window_size', None)
        # self.evaluate(experiment_path,
        #               time_ratio=time_ratio,
        #               postprocessing=postprocessing,
        #               threshold=threshold,
        #               window_size=window_size)

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    fire.Fire(Runner)
    