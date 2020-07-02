from utils.config import *
import logging
import os
import sys


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


"""  Default path setting """
DEFAULT_RAW_DIR = '/users4/ythou/Projects/TaskOrientedDialogue/data/FewShotNLU/RawData/'
DEFAULT_DATA_DIR = '/users4/ythou/Projects/TaskOrientedDialogue/data/FewShotNLU/Data/'
BERT_BASE_UNCASED = '/users4/ythou/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/'
BERT_BASE_UNCASED_VOCAB = '/users4/ythou/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/vocab.txt'


def define_args(parser, *args_builders):
    """ Set program args"""
    for args_builder in args_builders:
        parser = args_builder(parser)
    return parser


def basic_args(parser):
    group = parser.add_argument_group('Path')  # define path
    group.add_argument('--train_path', required=False, help='the path to the training file.')
    group.add_argument('--dev_path', required=False, help='the path to the validation file.')
    group.add_argument('--test_path', required=False, help='the path to the testing file.')
    group.add_argument("--eval_script", default='./scripts/conlleval.pl', help="The path to the evaluation script")
    group.add_argument("--bert_path", type=str, default=BERT_BASE_UNCASED, help="path to pretrained BERT")
    group.add_argument("--bert_vocab", type=str, default=BERT_BASE_UNCASED_VOCAB, help="path to BERT vocab file")
    group.add_argument('--output_dir', help='The dir to the output file, and to save model,eg: ./')
    group.add_argument("--saved_model_path", default='', help="path to the pre-trained model file")
    group.add_argument("--embedding_cache", type=str, default='/users4/ythou/Projects/Homework/ComputationalSemantic/.word_vectors_cache',
                       help="path to embedding cache dir. if use pytorch nlp, use this path to avoid downloading")

    group = parser.add_argument_group('Function')
    parser.add_argument("--task", default='sc', choices=['sl', 'sc'],
                        help="Task: sl:sequence labeling, sc:single label sent classify")
    group.add_argument('--allow_override', default=False, action='store_true', help='allow override experiment file')
    group.add_argument('--load_feature', default=False, action='store_true', help='load feature from file')
    group.add_argument('--save_feature', default=False, action='store_true', help='save feature to file')
    group.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    group.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    group.add_argument("--do_debug", default=False, action='store_true', help="debug model, only load few data.")
    group.add_argument("-doft", "--do_overfit_test", default=False, action='store_true', help="debug model, test/dev on train")
    group.add_argument("--verbose", default=False, action='store_true', help="Verbose logging")
    group.add_argument('--seed', type=int, default=42, help="the ultimate answer")

    group = parser.add_argument_group('Device')  # default to use all available GPUs
    group.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    group.add_argument("--no_cuda", default=False, action='store_true', help="Whether not to use CUDA when available")
    return parser


def preprocess_args(parser):
    group = parser.add_argument_group('Preprocess')
    group.add_argument("--sim_annotation", default='match', type=str,
                       choices=['match', 'BI-match', 'attention'], help="Define annotation of token similarity")
    group.add_argument("--label_wp", action='store_true',
                       help="For sequence label, use this to generate label for word piece, which is Abandon Now.")
    return parser


def train_args(parser):
    group = parser.add_argument_group('Train')
    group.add_argument("--restore_cpt", action='store_true', help="Restart training from a checkpoint ")
    group.add_argument("--cpt_per_epoch", default=2, type=int, help="The num of check points of each epoch")
    group.add_argument("--convergence_window", default=5000, type=int,
                       help="A observation window for early stop when model is in convergence, set <0 to disable")
    group.add_argument("--convergence_dev_num", default=5, type=int,
                       help="A observation window for early stop when model is in convergence, set <0 to disable")
    group.add_argument("--train_batch_size", default=2, type=int, help="Total batch size for training.")
    group.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    group.add_argument("--num_train_epochs", default=20, type=float,
                       help="Number of training epochs to perform.")
    group.add_argument("--warmup_proportion", default=0.1, type=float,
                       help="Proportion of training to perform linear learning rate warmup for. E.g.10%% of training.")
    group.add_argument("--eval_when_train", default=False, action='store_true',
                       help="Test model found new best model")

    group = parser.add_argument_group('SpaceOptimize')  # Optimize space usage
    group.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help="Number of updates steps to accumulate before performing a backward/update pass."
                            "Every time a variable is back propogated through,"
                            "the gradient will be accumulated instead of being replaced.")
    group.add_argument('--optimize_on_cpu', default=False, action='store_true',
                       help="Whether to perform optimization and keep the optimizer averages on CPU")
    group.add_argument('--fp16', default=False, action='store_true',
                       help="Whether to use 16-bit float precision instead of 32-bit")
    group.add_argument('--loss_scale', type=float, default=128,
                       help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    group.add_argument('-d_cpt', '--delete_checkpoint', default=False, action='store_true',
                       help="Only keep the best model to save disk space")

    group = parser.add_argument_group('PerformanceTricks')  # Training Tricks
    group.add_argument('--clip_grad', type=float, default=-1, help="clip grad, set < 0 to disable this")
    group.add_argument("--scheduler", default='linear_warmup', type=str, help="select pytorch scheduler for training",
                       choices=['linear_warmup', 'linear_decay'])
    group.add_argument('--decay_lr', type=float, default=0.5,
                       help="When choose linear_decay scheduler, rate of lr decay. ")
    group.add_argument('--decay_epoch_size', type=int, default=1,
                       help="When choose linear_decay scheduler, decay lr every decay_epoch_size")
    group.add_argument("--sampler_type", default='similar_len', choices=['similar_len', 'random'],
                       help="method to sample batch")

    # for few shot seq labeling model
    group = parser.add_argument_group('FewShotSetting')  # Training Tricks
    group.add_argument("--warmup_epoch", type=int, default=-1,
                       help="set > 0 to active warm up training. "
                            "Train model in two step: "
                            "1: fix bert part  "
                            "2: train entire model"
                            "(As we use new optimizer in 2nd stage, it also has restart effects. )")
    group.add_argument("--fix_embed_epoch", default=-1, type=int, help="Fix embedding for first x epochs.[abandon]")
    group.add_argument("--upper_lr", default=-1, type=float,
                       help="Use different LR for upper structure comparing to embedding LR. -1 to off it")
    group.add_argument("--no_embedder_grad", default=False, action='store_true', help="not perform grad on embedder")
    group.add_argument("--train_label_mask")

    return parser


def test_args(parser):
    group = parser.add_argument_group('Test')
    group.add_argument("--test_batch_size", default=2, type=int, help="Must same to few-shot batch size now")
    group.add_argument("--test_on_cpu", default=False, action='store_true', help="eval on cpu")

    return parser


def model_args(parser):
    group = parser.add_argument_group('Encoder')
    group.add_argument('--separate_prj', action='store_true', help='use different proj layer for support and test')
    group.add_argument("--projection_layer", default='none', type=str,
                       choices=['1-mlp', '2-mlp', 'mlp-relu', 'lstm', 'none'], help="select projection layer type")
    group.add_argument("--context_emb", default='bert', type=str,
                       choices=['bert', 'elmo', 'glove', 'raw', 'sep_bert', 'electra'],
                       help="select word representation type")
    group.add_argument("--similarity", default='dot', type=str,
                       choices=['cosine', 'dot', 'bi-affine', 'l2'], help="Metric for evaluating 2 tokens.")
    group.add_argument("--emb_dim", default=64, type=int, help="Embedding dimension for baseline")
    group.add_argument("--label_reps", default='sep', type=str,
                       choices=['cat', 'sep', 'sep_sum'], help="Method to represent label")
    group.add_argument("--use_schema", default=False, action='store_true',
                       help="(For MNet) Divide emission by each tag's token num in support set")

    group = parser.add_argument_group('Decoder')
    group.add_argument("--decoder", default='crf', type=str, choices=['crf', 'sms', 'rule', 'sc'],
                       help="decode method")

    # ===== emission layer setting =========
    group.add_argument("--emission", default='mnet', type=str,
                       choices=['mnet', 'rank', 'proto', 'proto_with_label', 'tapnet'],
                       help="Method for calculate emission score")
    group.add_argument("-e_nm", "--emission_normalizer", type=str, default='', choices=['softmax', 'norm', 'none'],
                       help="normalize emission into 1-0")
    group.add_argument("-e_scl", "--emission_scaler", type=str, default=None,
                       choices=['learn', 'fix', 'relu', 'exp', 'softmax', 'norm', 'none'],
                       help="method to scale emission and transition into 1-0")
    group.add_argument("--ems_scale_r", default=1, type=float, help="Scale transition to x times")
    # proto with label setting
    group.add_argument("-ple_nm", "--ple_normalizer", type=str, default='', choices=['softmax', 'norm', 'none'],
                       help="normalize scaled label embedding into 1-0")
    group.add_argument("-ple_scl", "--ple_scaler", type=str, default=None,
                       choices=['learn', 'fix', 'relu', 'exp', 'softmax', 'norm', 'none'],
                       help="method to scale label embedding into 1-0")
    group.add_argument("--ple_scale_r", default=1, type=float, help="Scale label embedding to x times")
    # tap net setting
    group.add_argument("--tap_random_init", default=False, action='store_true',
                       help="Set random init for label reps in tap-net")
    group.add_argument("--tap_random_init_r", default=1, type=float,
                       help="Set random init rate for label reps in tap-net")
    group.add_argument("--tap_mlp", default=False, action='store_true', help="Set MLP in tap-net")
    group.add_argument("--tap_mlp_out_dim", default=768, type=int, help="The dimension of MLP in tap-net")
    group.add_argument("--tap_proto", default=False, action='store_true',
                       help="choose use proto or label in projection space in tap-net method")
    group.add_argument("--tap_proto_r", default=1, type=float,
                       help="the rate of prototype in mixing with label reps")
    # Matching Network setting
    group.add_argument('-dbt', "--div_by_tag_num", default=False, action='store_true',
                       help="(For MNet) Divide emission by each tag's token num in support set")

    group.add_argument("--emb_log", default=False, action='store_true', help="Save embedding log in all emission step")

    # ===== decoding layer setting =======
    # CRF setting
    group.add_argument('--transition', default='learn',
                       choices=['merge', 'target', 'source', 'learn', 'none', 'learn_with_label'],
                       help='transition for target domain')
    group.add_argument("-t_nm", "--trans_normalizer", type=str, default='', choices=['softmax', 'norm', 'none'],
                       help="normalize back-off transition into 1-0")
    group.add_argument("-t_scl", "--trans_scaler", default=None,
                       choices=['learn', 'fix', 'relu', 'exp', 'softmax', 'norm', 'none'],
                       help='transition matrix scaler, such as re-scale the value to non-negative')

    group.add_argument('--backoff_init', default='rand', choices=['rand', 'fix'],
                       help='back-off transition initialization method')
    group.add_argument("--trans_r", default=1, type=float, help="Transition trade-off rate of src(=1) and tgt(=0)")
    group.add_argument("--trans_scale_r", default=1, type=float, help="Scale transition to x times")

    group.add_argument("-lt_nm", "--label_trans_normalizer", type=str, default='', choices=['softmax', 'norm', 'none'],
                       help="normalize transition FROM LABEL into 1-0")
    group.add_argument("-lt_scl", "--label_trans_scaler", default='fix', choices=['none', 'fix', 'learn'],
                       help='transition matrix FROM LABEL scaler, such as re-scale the value to non-negative')
    group.add_argument("--label_trans_scale_r", default=1, type=float, help="Scale transition FROM LABEL to x times")

    group.add_argument('-mk_tr', "--mask_transition", default=False, action='store_true',
                       help="Block out-of domain transitions.")
    group.add_argument("--add_transition_rules", default=False, action='store_true', help="Block invalid transitions.")

    group = parser.add_argument_group('Loss')
    group.add_argument("--loss_func", default='cross_entropy', type=str, choices=['mse', 'cross_entropy'],
                       help="Loss function for label prediction, when use crf, this factor is useless. ")

    group = parser.add_argument_group('Data')
    group.add_argument("--index_label", default=False, action="store_true", help="set the label as no meaning index")
    group.add_argument("--unused_label", default=False, action="store_true", help="set the label as bert [unusedID]")
    return parser


def option_check(opt):
    if opt.do_debug:
        if not opt.do_overfit_test:
            opt.num_train_epochs = 3
        opt.load_feature = False
        opt.save_feature = False
        opt.cpt_per_epoch = 1
        opt.allow_override = True

    if not(opt.local_rank == -1 or opt.no_cuda):
        if opt.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            opt.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)

    if not opt.do_train and not opt.do_predict:
        raise ValueError("At least one of 'do_train' or 'do_predict' must be True.")

    if os.path.exists(opt.output_dir) and os.listdir(opt.output_dir) and not opt.allow_override:
        raise ValueError("Output directory () already exists and is not empty.")

    if opt.do_train and not (opt.train_path and opt.dev_path):
        raise ValueError("If `do_train` is True, then `train_file` and dev file must be specified.")

    if opt.do_predict and not opt.test_path:
        raise ValueError("If `do_predict` is True, then `predict_file` must be specified.")

    if opt.gradient_accumulation_steps > opt.train_batch_size:
        raise ValueError('if split batch "gradient_accumulation_steps" must less than batch size')

    if opt.label_wp:
        raise ValueError('This function is kept in feature process but not support by current models.')
    return opt
