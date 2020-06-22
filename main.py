#!/usr/bin/env python
from typing import List, Tuple, Dict
import argparse, copy
import logging
import sys
import torch
import random
import os
import json
import pickle
# my staff
from utils.data_loader import FewShotRawDataLoader
from utils.preprocessor import FeatureConstructor, BertInputBuilder, FewShotOutputBuilder, make_dict, \
    save_feature, load_feature, make_preprocessor, make_label_mask, make_word_dict
from utils.opt import define_args, basic_args, train_args, test_args, preprocess_args, model_args, option_check
from utils.device_helper import prepare_model, set_device_environment
from utils.trainer import FewShotTrainer, SchemaFewShotTrainer, prepare_optimizer
from utils.tester import FewShotTester, SchemaFewShotTester, eval_check_points
from utils.model_helper import make_model, load_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_training_data_and_feature(opt, data_loader, preprocessor):
    """ prepare feature and data """
    if opt.load_feature:
        try:
            train_features, train_label2id, train_id2label = load_feature(opt.train_path.replace('.json', '.saved.pk'))
            dev_features, dev_label2id, dev_id2label = load_feature(opt.dev_path.replace('.json', '.saved.pk'))
        except FileNotFoundError:
            opt.load_feature, opt.save_feature = False, True  # Not a saved feature file yet, make it
            train_features, train_label2id, train_id2label, dev_features, dev_label2id, dev_id2label =\
                get_training_data_and_feature(opt, data_loader, preprocessor)
            opt.load_feature, opt.save_feature = True, False  # restore option
    else:
        train_examples, train_max_len, train_max_support_size = data_loader.load_data(path=opt.train_path)
        dev_examples, dev_max_len, dev_max_support_size = data_loader.load_data(path=opt.dev_path)
        train_label2id, train_id2label = make_dict(opt, train_examples)
        dev_label2id, dev_id2label = make_dict(opt, dev_examples)
        logger.info(' Finish train dev prepare dict ')
        train_features = preprocessor.construct_feature(
            train_examples, train_max_support_size, train_label2id, train_id2label)
        dev_features = preprocessor.construct_feature(dev_examples, dev_max_support_size, dev_label2id, dev_id2label)
        logger.info(' Finish prepare train dev features ')

        if opt.save_feature:
            save_feature(opt.train_path.replace('.json', '.saved.pk'), train_features, train_label2id, train_id2label)
            save_feature(opt.dev_path.replace('.json', '.saved.pk'), dev_features, dev_label2id, dev_id2label)
    return train_features, train_label2id, train_id2label, dev_features, dev_label2id, dev_id2label


def get_testing_data_feature(opt, data_loader, preprocessor):
    """ prepare feature and data """
    if opt.load_feature:
        try:
            test_features, test_label2id, test_id2label = load_feature(opt.test_path.replace('.json', '.saved.pk'))
        except FileNotFoundError:
            opt.load_feature, opt.save_feature = False, True  # Not a saved feature file yet, make it
            test_features, test_label2id, test_id2label = get_testing_data_feature(opt, data_loader, preprocessor)
            opt.load_feature, opt.save_feature = True, False  # restore option
    else:
        test_examples, test_max_len, test_max_support_size = data_loader.load_data(path=opt.test_path)
        test_label2id, test_id2label = make_dict(opt, test_examples)
        logger.info(' Finish prepare test dict')
        test_features = preprocessor.construct_feature(
            test_examples, test_max_support_size, test_label2id, test_id2label)
        logger.info(' Finish prepare test feature')
        if opt.save_feature:
            save_feature(opt.test_path.replace('.json', '.saved.pk'), test_features, test_label2id, test_id2label)
    return test_features, test_label2id, test_id2label


def main():
    """ to start the experiment """
    ''' set option '''
    parser = argparse.ArgumentParser()
    parser = define_args(parser, basic_args, train_args, test_args, preprocess_args, model_args)
    opt = parser.parse_args()
    print('Args:\n', json.dumps(vars(opt), indent=2))
    opt = option_check(opt)

    ''' device & environment '''
    device, n_gpu = set_device_environment(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    logger.info("Environment: device {}, n_gpu {}".format(device, n_gpu))

    ''' data & feature '''
    data_loader = FewShotRawDataLoader(opt)
    preprocessor = make_preprocessor(opt)
    if opt.do_train:
        train_features, train_label2id, train_id2label, dev_features, dev_label2id, dev_id2label = \
            get_training_data_and_feature(opt, data_loader, preprocessor)

        if opt.mask_transition and opt.task == 'sl':
            opt.train_label_mask = make_label_mask(opt, opt.train_path, train_label2id)
            opt.dev_label_mask = make_label_mask(opt, opt.dev_path, dev_label2id)
    else:
        train_features, train_label2id, train_id2label, dev_features, dev_label2id, dev_id2label = [None] * 6
        if opt.mask_transition and opt.task == 'sl':
            opt.train_label_mask = None
            opt.dev_label_mask = None
    if opt.do_predict:
        test_features, test_label2id, test_id2label = get_testing_data_feature(opt, data_loader, preprocessor)
        if opt.mask_transition and opt.task == 'sl':
            opt.test_label_mask = make_label_mask(opt, opt.test_path, test_label2id)
    else:
        test_features, test_label2id, test_id2label = [None] * 3
        if opt.mask_transition and opt.task == 'sl':
            opt.test_label_mask = None

    ''' over fitting test '''
    if opt.do_overfit_test:
        test_features, test_label2id, test_id2label = train_features, train_label2id, train_id2label
        dev_features, dev_label2id, dev_id2label = train_features, train_label2id, train_id2label

    ''' select training & testing mode '''
    trainer_class = SchemaFewShotTrainer if opt.use_schema else FewShotTrainer
    tester_class = SchemaFewShotTester if opt.use_schema else FewShotTester

    ''' training '''
    best_model = None
    if opt.do_train:
        logger.info("***** Perform training *****")
        if opt.restore_cpt:  # restart training from a check point.
            training_model = load_model(opt.saved_model_path)  # restore optimizer param is not support now.
            opt = training_model.opt
            opt.warmup_epoch = -1
        else:
            training_model = make_model(opt, config={'num_tags': len(train_label2id)})
        training_model = prepare_model(opt, training_model, device, n_gpu)
        if opt.mask_transition and opt.task == 'sl':
            training_model.label_mask = opt.train_label_mask.to(device)
        # prepare a set of name subseuqence/mark to use different learning rate for part of params
        upper_structures = [
            'backoff', 'scale_rate', 'f_theta', 'phi', 'start_reps', 'end_reps', 'biaffine']
        param_to_optimize, optimizer, scheduler = prepare_optimizer(
            opt, training_model, len(train_features), upper_structures)
        tester = tester_class(opt, device, n_gpu)
        trainer = trainer_class(opt, optimizer, scheduler, param_to_optimize, device, n_gpu, tester=tester)
        if opt.warmup_epoch > 0:
            training_model.no_embedder_grad = True
            stage_1_param_to_optimize, stage_1_optimizer, stage_1_scheduler = prepare_optimizer(
                opt, training_model, len(train_features), upper_structures)
            stage_1_trainer = trainer_class(opt, stage_1_optimizer, stage_1_scheduler, stage_1_param_to_optimize, device, n_gpu, tester=None)
            trained_model, best_dev_score, test_score = stage_1_trainer.do_train(
                training_model, train_features, opt.warmup_epoch)
            training_model = trained_model
            training_model.no_embedder_grad = False
            print('========== Warmup training finished! ==========')
        trained_model, best_dev_score, test_score = trainer.do_train(
            training_model, train_features, opt.num_train_epochs,
            dev_features, dev_id2label, test_features, test_id2label, best_dev_score_now=0)

        # decide the best model
        if not opt.eval_when_train:  # select best among check points
            best_model, best_score, test_score_then = trainer.select_model_from_check_point(
                train_id2label, dev_features, dev_id2label, test_features, test_id2label, rm_cpt=opt.delete_checkpoint)
        else:  # best model is selected during training
            best_model = trained_model
        logger.info('dev:{}, test:{}'.format(best_dev_score, test_score))
        print('dev:{}, test:{}'.format(best_dev_score, test_score))

    ''' testing '''
    if opt.do_predict:
        logger.info("***** Perform testing *****")
        print("***** Perform testing *****")
        tester = tester_class(opt, device, n_gpu)
        if not best_model:  # no trained model load it from disk.
            if not opt.saved_model_path or not os.path.exists(opt.saved_model_path):
                raise ValueError("No model trained and no trained model file given (or not exist)")
            if os.path.isdir(opt.saved_model_path):  # eval a list of checkpoints
                max_score = eval_check_points(opt, tester, test_features, test_id2label, device)
                print('best check points scores:{}'.format(max_score))
                exit(0)
            else:
                best_model = load_model(opt.saved_model_path)

        ''' test the best model '''
        testing_model = tester.clone_model(best_model, test_id2label)  # copy reusable params
        if opt.mask_transition and opt.task == 'sl':
            testing_model.label_mask = opt.test_label_mask.to(device)
        test_score = tester.do_test(testing_model, test_features, test_id2label, log_mark='test_pred')
        logger.info('test:{}'.format(test_score))
        print('test:{}'.format(test_score))


if __name__ == "__main__":
    main()
