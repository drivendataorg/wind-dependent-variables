import numpy as np

import torch
from torch import nn
import pretrainedmodels
from torchvision import transforms
from torch.utils.data import DataLoader

from src.utils.misc import AttrDict
from src.data.storms_dataset import get_storms_df, StormsDatasetSequence, DATASET_MEAN, DATASET_STD
from src.data.data_feeders import DataFeed
from src.pytorch.utils import flip


MODEL_TYPE = 'L1A'
MODEL_VERSION = 'A'
TRAINING_VERSION = '6'  # vgg11_gru_s24g10r1B - CS061bn24g10r1

args = AttrDict({})

# Data
args.update({
    'sequence_gap': 1.0,  # hours
    'sequence_length': 8*3,
})
# Train ALL
args.update({
    'all_train_epochs': None,
    'all_train_lr_step': None,
})
# Model
args.update({
    'basemodel_name': 'vgg11_bn',
    'pretrained': 'imagenet', # 'imagenet' or 'imagenet+background'
    'freeze_basemodel': False,  # Indicate first layer to train, True for freeze all, False none
    'reset_weights': False,
})
# Transformations
SIZE = 224
args.update({
    'transformations': {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation((0,360), expand=True),
            transforms.CenterCrop((366, 366)),
            transforms.Resize(SIZE),
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(SIZE),
        ]),
        'fold': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(SIZE),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(SIZE),
        ]),
    }
})

# Detect if we have a GPU available
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
PARALLEL_TRAINING = False
NB_GPU = max(1, torch.cuda.device_count()) if PARALLEL_TRAINING else 1

# Execution
TRAIN_BATCH_SIZE = 8 * NB_GPU
VIRTUAL_BATCH_SIZE = None  # Accumulate gradients, Int or None. It may not work properly
args.update({
    'max_train_epochs': 40,
    'train_batch_size': TRAIN_BATCH_SIZE,
    'valid_batch_size': TRAIN_BATCH_SIZE,
    'test_batch_size': TRAIN_BATCH_SIZE * 1,
    'virtual_batch_size': VIRTUAL_BATCH_SIZE,
    'num_workers': 8 * NB_GPU,
    'parallel_training': PARALLEL_TRAINING,
})

# Model
args.update({
    'ouput_name': f"{MODEL_TYPE}_{MODEL_VERSION}{TRAINING_VERSION}"
})

ARGS = args

BASEMODEL_FEATURES = {
    'vgg11_bn': (25088, lambda model: nn.Sequential(*list(model.children())[1:-7], nn.Flatten())),
}


def get_dset(path_to_data, args=ARGS):

    print('-' * 40)
    print("READING DATASET")

    # Get dataset
    dset_df = get_storms_df(path_to_data)
    print(f"Dataset shape: {dset_df.shape})")
    print(f"Training: {dset_df.train.sum()}| Testing: {dset_df.test.sum()}")

    return dset_df


def _generate_data_feeders(df_dict, load_function, transformations, preprocess, phases, args_dict):
    return {s1: DataFeed(df_dict[s1], load_function[s1],
                         x_transform_func=transformations[s1],
                         x_preprocess_func=preprocess,
                         y_preprocess_func=None,
                         **args_dict[s1]) for s1 in phases}


def _generate_data_loaders(date_feeder, phases, args_dict):
    return {s1: DataLoader(date_feeder[s1], **args_dict[s1]) for s1 in phases}


def _worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_dataloaders(path_to_data, datasets, args=ARGS):
    train_stages = datasets.keys()

    ## Data reader classes
    storms_data_reader = {
        'train': StormsDatasetSequence(datasets['train'], nb_imgs=args.sequence_length, gap=args.sequence_gap,
                                       missing='black', max_block_size=5, return_all_labels=True, reduce_all_labels=3),
        'valid': StormsDatasetSequence(datasets['valid'],nb_imgs=args.sequence_length,  gap=args.sequence_gap,
                                       missing='black'),
        'fold': StormsDatasetSequence(datasets['fold'], nb_imgs=args.sequence_length, gap=args.sequence_gap,
                                      missing='black'),
        'test': StormsDatasetSequence(datasets['test'], nb_imgs=args.sequence_length, gap=args.sequence_gap,
                                      missing='black'),
    }

    ## Load images function
    load_function = {
        'train': lambda x: storms_data_reader['train'].read(x),
        'valid': lambda x: storms_data_reader['valid'].read(x),
        'fold': lambda x: storms_data_reader['fold'].read(x),
        'test': lambda x: storms_data_reader['test'].read(x),
    }

    # Data Feeders
    dsmean, dsstd = DATASET_MEAN / 255, DATASET_STD / 255
    img_pre_process_tt = transforms.Compose([
        transforms.Normalize(mean=dsmean, std=dsstd)
    ])

    def preprocess(imgs):
        imgs = [img_pre_process_tt(s) for s in imgs]
        return torch.stack(imgs).to(torch.float32)

    transformations = args.transformations
    data_feeder_args = {
        'train': {'scale_df': 4},
        'valid': {},
        'fold': {},
        'test': {'predict': True},
    }

    # Data Feeders
    dts = {s1:storms_data_reader[s1].get_iter_df() for s1 in datasets.keys()}
    data_feeders = _generate_data_feeders(dts, load_function, transformations, preprocess, train_stages,
                                          data_feeder_args)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    ## Data Loader
    data_loader_args = {
        "train": {'batch_size': args.train_batch_size, 'shuffle': True, 'worker_init_fn': worker_init_fn, },
        "valid": {'batch_size': args.valid_batch_size, 'shuffle': False, 'worker_init_fn': worker_init_fn, },
        "fold": {'batch_size': args.valid_batch_size, 'shuffle': False, 'worker_init_fn': worker_init_fn, },
        "test": {'batch_size': args.test_batch_size, 'shuffle': False, 'worker_init_fn': worker_init_fn, },
    }
    if 'cuda' in DEVICE:
        for phase, dct in data_loader_args.items():
            dct.update({'pin_memory': True, 'num_workers': args.num_workers})
    # Data Loaders
    data_loaders = _generate_data_loaders(data_feeders, train_stages, data_loader_args)

    return data_loaders


class NNModel(nn.Module):
    def __init__(self, basemodel, num_ftrs, num_channels):
        super(NNModel, self).__init__()

        self.features = basemodel
        self.num_ftrs = num_ftrs
        self.num_channels = num_channels

        input_size = num_ftrs
        hidden_size = 1024
        num_layers = 1
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.regressor = nn.Sequential(*[
            nn.Linear(hidden_size*num_layers, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 1),
        ])

        self.output_predictions = False

    def forward(self, x):

        bs = x.size()[0]
        seq = x.size()[1]
        ss = seq // 3
        sx = x.size()[-2]
        sy = x.size()[-1]
        x = x.reshape(bs, ss, 3, sx, sy)
        x = x.view(-1, 3, sx, sy)

        x = self.features(x)
        x = x.view(bs, ss, -1)
        x = flip(x, 1)  # [first...last]

        output, hn = self.gru(x)
        output = flip(output, 1)  # [last...first]

        if self.training:
            x = output.view(bs * ss, -1)
            x = self.regressor(x)
            x = x.view(bs, ss)
            return x
        else:
            x = hn.transpose(0, 1).view(bs, -1)
            x = self.regressor(x)

        if self.output_predictions:
            return x
        else:
            return torch.squeeze(x)


def _get_model(nb_classes, args=ARGS):
    print('-' * 80)
    print(f"LOADING MODEL")
    print(f"Using device: {DEVICE}")
    # Get basemodel
    num_classes = 1001 if args.pretrained == 'imagenet+background' else 1000
    basemodel = pretrainedmodels.__dict__[args.basemodel_name](num_classes=num_classes, pretrained=args.pretrained)
    basemodel = BASEMODEL_FEATURES[args.basemodel_name][1](basemodel)
    num_ftrs = BASEMODEL_FEATURES[args.basemodel_name][0]

    if args.freeze_basemodel is not None and args.freeze_basemodel is not False:
        if args.freeze_basemodel is True:
            print("Freezing base model")
            for param in basemodel.parameters():
                param.requires_grad = False
        else:
            print("Params to learn:")
            train = False
            for name, param in basemodel.named_parameters():
                if args.freeze_basemodel in name or train:
                    param.requires_grad = True
                    train = True
                    print("\t", name)
                else:
                    param.requires_grad = False
    
    if args.reset_weights:
        print("Reseting weights...")
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        basemodel.apply(weight_reset)

    model = NNModel(basemodel, num_ftrs, args.sequence_length)

    data_parallel = False
    if args.parallel_training:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            data_parallel = True

    return model


def get_learner(nb_classes, all=False, args=ARGS):

    model = _get_model(nb_classes, args=args)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    from src.metrics import pytorch_metrics
    criterion = pytorch_metrics.RMSELoss()
    metrics = [pytorch_metrics.RMSELoss(round=True), ]

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    if all:
        from src.pytorch.schedulers import StepLR
        scheduler = StepLR(optimizer, step_size=args.all_train_lr_step, gamma=0.1, min_lr=1e-5)
        from src.pytorch.early_stoppers import EarlyStopping
        early_stopper = None
    else:
        #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True, threshold=0.001,
        #                              min_lr=1e-5)
        from src.pytorch.schedulers import StepLR
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1, min_lr=1e-5)
        from src.pytorch.early_stoppers import EarlyStopping
        early_stopper = EarlyStopping(patience=6, verbose=True, delta=0.001, save_model_path=None, wait=15)

    from src.pytorch.wrappers import PyTorchNN_vA as PyTorchNN
    pytorchmodel = PyTorchNN(model, optimizer, criterion, metrics, scheduler, early_stopper, device=DEVICE,
                             virtual_batch_size=args.virtual_batch_size)

    return pytorchmodel


def save_model(filename, learner):
    print("-" * 80)
    print("SAVE MODEL")
    # Generate checkpoint to save
    checkpoint = learner.model_checkpoint.copy()
    print(f"Saving model to file: {filename}")
    torch.save(checkpoint, filename)


def load_model(filename, nb_classes=None, load_learner=False):
    print("-" * 80)
    print("LOAD MODEL")
    # Loading model
    print(f"Loading model from file: {filename}")
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage.cuda(0))
    # Generate model
    model = _get_model(nb_classes=nb_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Load learner
    if load_learner:
        pytorchmodel = get_learner(model=model)
    else:
        from src.pytorch.wrappers import PyTorchNN_vA as PyTorchNN
        pytorchmodel = PyTorchNN(model, device=DEVICE)

    return pytorchmodel