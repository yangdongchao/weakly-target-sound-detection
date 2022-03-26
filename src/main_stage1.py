# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/17 16:42
# @Author  : dongchao yang
# @File    : main_stage1.py
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
from torch.utils.tensorboard import SummaryWriter
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
        # inputs, ,, time, embedding, filenames,events = batch
        inputs, clip_level_targets, embedding, filenames,events = batch
        # print(events)
        # print(frame_level_time[:,0])
        # assert 1==2
        inputs = convert_tensor(inputs, device=DEVICE, non_blocking=True)
        clip_level_targets = convert_tensor(clip_level_targets.float(), device=DEVICE, non_blocking=True)
        embedding = convert_tensor(embedding, device=DEVICE, non_blocking=True)
        w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time = model(inputs,embedding)
        return w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time, clip_level_targets
    
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
        writter = SummaryWriter(os.path.join(outputdir, 'train_loss_log'))
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
        trainloader = dataset.getdataloader(
            config_parameters['train_data'], # feature path
            config_parameters['spk_emb_file_path'],
            transform=transform,
            batch_size=config_parameters['batch_size'],
            num_workers=config_parameters['num_workers'],**sampling_kwargs)

        cvdataloader = dataset.getdataloader(
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
            #optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learningRate)
            optimizer = getattr(torch.optim,config_parameters['optimizer'],)(filter(lambda p: p.requires_grad, model.parameters()), **config_parameters['optimizer_args']) # 加载 optimizer

        utils.pprint_dict(optimizer, logger.info, formatter='pretty')
        utils.pprint_dict(model, logger.info, formatter='pretty')
        if DEVICE.type != 'cpu' and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model) # 同时使用多个GPU
        criterion = getattr(losses, config_parameters['loss'])().to(DEVICE)
        criterion_dis = getattr(losses, config_parameters['loss_dis'])().to(DEVICE)
        def _train_batch(_, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time, clip_level_target = self._forward(model, batch)
                # print('decision ',decision.shape)
                # print('frame_level_target ',frame_level_target.shape)
                # assert 1==2
                loss_sed = criterion(w_clip_decision[:,0],clip_level_target[:,0])
                loss_dis = criterion_dis(w_out,s_out)
                # print('loss_sed ',loss_sed)
                # print('loss_dis ',loss_dis)
                # assert 1==2
                loss = loss_sed + 5*loss_dis
                loss.backward()
                # Single loss
                optimizer.step()
                writter.add_scalar('tot_loss', loss.cpu().item(), train_engine.state.epoch)
                writter.add_scalar('loss_sed', loss_sed.cpu().item(), train_engine.state.epoch)
                writter.add_scalar('loss_dis', loss_dis.cpu().item(), train_engine.state.epoch)
                return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                return self._forward(model, batch)

        def thresholded_output_transform(output):
            w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time, clip_level_target = output
            #y_pred = torch.round(decision) # 将输入input张量每个元素舍入到最近的整数
            y_pred  = w_clip_decision > 0.5
            y_pred = y_pred.long()
            return y_pred[:,0], clip_level_target[:,0]

        precision = Precision(thresholded_output_transform, average=False)
        recall = Recall(thresholded_output_transform, average=False)
        f1_score = (precision * recall * 2 / (precision + recall)).mean()
        metrics = {
            'Loss': losses.Loss_stage1(criterion),  #reimplementation of Loss, supports 3 way loss 
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
        dataloader = dataset.getdataloader(
            config_parameters['test_data'],
            config_parameters['spk_emb_file_path'],
            batch_size=1, shuffle=False)
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
        # config_parameters['metadata'] = '/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/mata_eval_choose.tsv'
        # config_parameters['label'] = '/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/strong_eval_choose_psds.tsv'
        # config_parameters['dtc_threshold'] = 0.5
        # config_parameters['gtc_threshold'] = 0.5
        # config_parameters['cttc_threshold'] = 0.3
        # config_parameters['alpha_ct'] = 0.0
        # config_parameters['alpha_st'] = 1
        # config_parameters['max_efpr'] = 100
        # # psds eval procss
        # psds = get_eval_score(os.path.join(experiment_path,'psds_prediction'),
        #                             config_parameters['label'],config_parameters['metadata'],config_parameters,log)
        # print(f"\nPSD-Score: {psds.value:.5f}")
        # assert 1==2
        with torch.no_grad():
            for batch in tqdm(dataloader, unit='file', leave=False): # dataloard 加载了弱标签
                inputs, clip_level_target, embedding, filenames,events = batch
                w_decision_time,w_clip_decision,w_decision_up,w_out,s_out,s_decision_time, clip_level_target = self._forward(model, batch) # 
                # mAP_tar.append(frame_level_target.detach().cpu().numpy().squeeze(0))
                # mAP_pred.append(decision.detach().cpu().numpy().squeeze(0))
                frame_num = w_decision_up.shape[1]
                pred = w_decision_up.detach().cpu().numpy()
                pred = pred[:,:,0]
                # print(pred.shape)
                if postprocessing == 'median':
                    if threshold is None:
                        thres = config_parameters['thres']
                    else:
                        thres = threshold
                    if window_size is None:
                        window_size = 1
                    filtered_pred = utils.median_filter(pred, window_size=window_size, threshold=thres)
                    for index_k in range(filtered_pred.shape[0]):
                        decoded_pred = []
                        decoded_pred_ = utils.decode_with_timestamps(events[index_k],filtered_pred[index_k,:])
                        if len(decoded_pred_) == 0: # neg deal
                            decoded_pred_.append((events[index_k],0,0))
                        decoded_pred.append(decoded_pred_)
                        for num_batch in range(len(decoded_pred)): # when we test our model,the batch_size is 1
                            #print('len(decoded_pred) ',len(decoded_pred))
                            filename = filenames[num_batch]
                            #print('filename ',filenames[num_batch])
                            cur_pred = pred[num_batch]
                            # Save each frame output, for later visualization
                            label_prediction = decoded_pred[num_batch] # frame predict
                            # print(label_prediction)
                            for event_label, onset, offset in label_prediction:
                                time_predictions.append({
                                    'filename': filename,
                                    'onset': onset,
                                    'offset': offset,
                                    'event_label': event_label}) # get real predict results,including event_label,onset,offset
                        # psds eval
                        if psds_eval_:
                            for i, th in enumerate(np.arange(0.01, 1.01, 0.01)):
                                thres = th
                                window_size = 1
                                filtered_pred = utils.median_filter(pred, window_size=window_size, threshold=thres)
                                for index_k in range(filtered_pred.shape[0]):
                                    decoded_pred = []
                                    decoded_pred_ = utils.decode_with_timestamps(events[index_k],filtered_pred[index_k,:])
                                    if len(decoded_pred_) == 0: # neg deal
                                        decoded_pred_.append((events[index_k],0,0))
                                    decoded_pred.append(decoded_pred_)
                                    # print(decoded_pred)
                                    for num_batch in range(len(decoded_pred)): # when we test our model,the batch_size is 1
                                        filename = filenames[num_batch]
                                        label_prediction = decoded_pred[num_batch] # frame predict                    
                                        for event_label, onset, offset in label_prediction:
                                            psds_time_predictions[i].append({
                                                'filename': filename,
                                                'onset': onset,
                                                'offset': offset,
                                                'event_label': event_label})
                else:
                    # Double thresholding as described in
                    # https://arxiv.org/abs/1904.03841
                    if threshold is None:
                        hi_thres, low_thres = (0.75, 0.2) # i change 0.75 to 0.7
                    else:
                        hi_thres, low_thres = threshold
                    filtered_pred = utils.double_threshold(pred, high_thres=hi_thres, low_thres=low_thres)
                    decoded_pred = utils.decode_with_timestamps(events[0],filtered_pred)
        time_ratio = 10. / frame_num
        assert len(time_predictions) > 0, "No outputs, lower threshold?"
        pred_df = pd.DataFrame(time_predictions, columns=['filename', 'onset', 'offset','event_label']) # it store the happen event and its time information
        pred_df = utils.predictions_to_time(pred_df, ratio=time_ratio) # transform the number of frame to real time
        test_data_filename = os.path.splitext(os.path.basename(config_parameters['label']))[0]
        if pred_file: # it name is hard_predictions...
            pred_df.to_csv(os.path.join(experiment_path, pred_file.format(test_data_filename)),index=False, sep="\t")
        if sed_eval:
            event_result, segment_result = metrics.compute_metrics(
                strong_labels_df, pred_df, time_resolution=0.2) # calculate f1
            print("Event Based Results:\n{}".format(event_result))
            event_results_dict = event_result.results_class_wise_metrics()
            class_wise_results_df = pd.DataFrame().from_dict({
                f: event_results_dict[f]['f_measure']
                for f in event_results_dict.keys()}).T
            class_wise_results_df.to_csv(os.path.join(
                experiment_path, class_result_file.format(test_data_filename)), sep='\t')
            print("Class wise F1-Macro:\n{}".format(
                tabulate(class_wise_results_df, headers='keys', tablefmt='github')))
            if event_file:
                with open(os.path.join(experiment_path,
                          event_file.format(test_data_filename)), 'w') as wp:
                    wp.write(event_result.__str__())
            print("=" * 100)
            print(segment_result)
            if segment_file:
                with open(os.path.join(experiment_path,
                          segment_file.format(test_data_filename)), 'w') as wp:
                    wp.write(segment_result.__str__())
            event_based_results = pd.DataFrame(
                event_result.results_class_wise_average_metrics()['f_measure'],
                index=['event_based'])
            segment_based_results = pd.DataFrame(
                segment_result.results_class_wise_average_metrics()
                ['f_measure'], index=['segment_based'])
            result_quick_report = pd.concat((event_based_results, segment_based_results))
            # Add two columns
            with open(os.path.join(experiment_path, 'quick_report_{}.md'.format(test_data_filename)),'w') as wp:
                print(tabulate(result_quick_report, headers='keys', tablefmt='github'), file=wp)
            print("Quick Report: \n{}".format(tabulate(result_quick_report, headers='keys', tablefmt='github')))
        
        if psds_eval_:
            if not os.path.exists(os.path.join(experiment_path,'psds_prediction')):
                os.mkdir(os.path.join(experiment_path,'psds_prediction'))
            for i, th in enumerate(np.arange(0.01, 1.01, 0.01)):
                pred_df_tmp = pd.DataFrame(psds_time_predictions[i], columns=['filename', 'onset', 'offset','event_label']) # it store the happen event and its time information
                pred_df_tmp = utils.predictions_to_time(pred_df_tmp, ratio=time_ratio) # transform the number of frame to real time
                save_name = str(int(th*100)) +'.tsv'
                pred_df_tmp.to_csv(os.path.join(experiment_path,'psds_prediction',save_name), index=False, sep="\t")
            config_parameters['metadata'] = '/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/mata_eval_choose.tsv'
            config_parameters['label'] = '/apdcephfs/share_1316500/donchaoyang/code2/data/choose_class/strong_eval_choose_psds.tsv'
            config_parameters['dtc_threshold'] = 0.7
            config_parameters['gtc_threshold'] = 0.7
            config_parameters['cttc_threshold'] = 0.3
            config_parameters['alpha_ct'] = 0.0
            config_parameters['alpha_st'] = 1
            config_parameters['max_efpr'] = 100
            psds = get_eval_score(os.path.join(experiment_path,'psds_prediction'),
                                        config_parameters['label'],config_parameters['metadata'],config_parameters,log)
            print(f"\nPSD-Score: {psds.value:.5f}")

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
        model = getattr(models, config_param['model'])(config_param,inputdim=64,
                        outputdim=2, **config_param['model_args'])
        model.load_state_dict(model_parameters)
        model.to(DEVICE)
        time_ratio = 10.0/500
        config_parameters = utils.parse_config_or_kwargs(config, **kwargs)
        threshold = config_parameters.get('threshold', None)
        postprocessing = config_parameters.get('postprocessing', 'double')
        window_size = config_parameters.get('window_size', None)
        self.evaluate(experiment_path,
                      time_ratio=time_ratio,
                      postprocessing=postprocessing,
                      threshold=threshold,
                      window_size=window_size)

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    fire.Fire(Runner)
    