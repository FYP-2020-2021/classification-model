import argparse

from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import L2
from tensorflow.keras.regularizers import L1
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adamax
from models.bilstm import BiLSTM
from models.bigru_attention import BiGRUAttention
from models.han import HAN
from data_manager import DataManager

model_names = ['bilstm', 'bigruattention', 'han', 'bert']
regularizers = ['l1', 'l2', 'l1l2']
optimizers = ['adam', 'adagrad', 'rmsprop', 'sgd', 'adamax']
loss = ['binary_crossentropy', 'categorical_crossentropy', 'mean_squared_error', 
        'sparse_categorical_crossentropy']
metrics = ['accuracy', 'binary_accuracy', 'binary_crossentropy', 
           'categorical_accuracy', 'sparse_categorical_accuracy', 
           'sparse_top_k_categorical_accuracy', 'top_k_categorical_accuracy']

#
# parse command-line arguments
#
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model-dir', type=str, default='', 
				help='path to desired output directory for saving model '
					'checkpoints (default: current directory)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='bilstm',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: bilstm)')
parser.add_argument('--encoding', default='utf8', type=str, 
                    help='encoding of the dataset texts (default: utf8')
# parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
#                     help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=35, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 8), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='show verbose when loading dataset (default: False)')
parser.add_argument('--resume', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
#                     help='use pre-trained model')
parser.add_argument('--train-ratio', default=0.9, type=float,
                    help='ratio between train and validation set (default: 0.9)')
parser.add_argument('--l1-regularizer-factor', default=0.01, type=float,
                    help='L1 regularizer penalty. (default: 0.01)')
parser.add_argument('--l2-regularizer-factor', default=0.01, type=float,
                    help='L2 regularizer penalty. (default: 0.01)')
parser.add_argument('--dropout', default=0.3, type=float,
                    help='dropout rate. (default: 0.3)')
parser.add_argument('--regularizer', default='l2', type=str,
                    choices=regularizers,
                    help='regularizers: ' +
                        ' | '.join(regularizers) +
                        ' (default: l2)')
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=optimizers,
                    help='optimizers: ' +
                        ' | '.join(optimizers) +
                        ' (default: adam)')
parser.add_argument('--nesterov', default=False, action='store_true',
                    help='SGD Nesterov. (default: False)')
parser.add_argument('--loss', default='sparse_categorical_crossentropy', type=str, 
                    choices=loss,
                    help='loss functions: ' +
                        ' | '.join(loss) +
                        ' (default: sparse_categorical_crossentropy)')
parser.add_argument('--metrics', default='sparse_categorical_accuracy', nargs='?', type=str,
                    choices=metrics,
                    help='performance metrics: ' +
                        ' | '.join(metrics) +
                        ' (default: sparse_categorical_accuracy)')
parser.add_argument('--state-sizes', default=[128, 256], type=int, nargs='?',
                    help='the hidden state sizes of the RNN layers, each element represents the hidden size of an RNN layer. (default: [128, 256]')
parser.add_argument('--embed-size', default=200, type=int, 
                    help='embedding layer size of the model.')
parser.add_argument('--cvl', default=100, type=int, 
                    help='the size of the hidden context vector. If set to 1 this layer reduces to a standard attention layer. (default: 100)')

args = parser.parse_args()



dm = DataManager(args.data, args.train_ratio, verbose=args.verbose, encoding=args.encoding)

# input_string = ['A text editor for Chrome OS and Chrome.\n' + \
# 'Text.app is a simple text editor for Chrome OS and Chrome. It\'s fast, lets you open multiple files at once, has syntax highlighting, and saves to Google Drive on Chrome OS.\n' + \
# '\n' + \
# 'File bugs:\n' + \
# 'https://github.com/GoogleChrome/text-app/issues\n' + \
# '\n' + \
# 'Version 0.5.186\n' + \
# '- Added screenreader mode to settings. Flipping this on turns off syntax highlighting, smart tab, line numbers and various other visual settings but greatly improves screenreader behavior.\n' + \
# '- Removed option to turn on analytics, removed all analytics code.\n' + \
# '- Added support for files with "xht", "xhtm" and "xhtml" extensions.\n' + \
# '- Removed behavior where doing control + s while caps lock was on would trigger a save as operation instead of a save operation.']

# tensor = dm.predict_preprocess(input_string)
# print(tensor)

if args.optimizer == 'adam':
    optimizer = Adam(args.lr)
elif args.optimizer == 'rmsprop':
    optimizer = RMSprop(args.lr, momentum=args.momentum)
elif args.optimizer == 'adagrad':
    optimizer = Adagrad(args.lr)
elif args.optimizer == 'adamax':
    optimizer = Adamax(args.lr)
elif args.optimizer == 'sgd':
    optimizer = SGD(args.lr, args.momentum, args.nesterov)

if args.regularizer == 'l1':
    regularizer = L1(args.l1_regularizer_factor)
elif args.regularizer == 'l2':
    regularizer = L2(args.l2_regularizer_factor)
elif args.regularizer == 'l1l2':
    regularizer = L1L2(l1=args.l1_regularizer_factor, l2=args.l2_regularizer_factor)

model = None
if args.arch == 'bilstm':
    model = BiLSTM(dm, regularizers=regularizer, epochs=args.epochs, dropout=args.dropout,
                            batch_size=args.batch_size, optimizer=optimizer,
                            checkpoint_path=args.model_dir, embed_size=args.embed_size,
                            loss=args.loss, metrics=args.metrics, state_sizes=args.state_sizes)
elif args.arch == 'bigruattention':
    model = BiGRUAttention(args.cvl, dm, regularizers=regularizer, epochs=args.epochs, dropout=args.dropout,
                            batch_size=args.batch_size, optimizer=optimizer,
                            checkpoint_path=args.model_dir, embed_size=args.embed_size,
                            loss=args.loss, metrics=args.metrics, state_sizes=args.state_sizes)
elif args.arch == 'han':
    model = HAN(args.cvl, dm, regularizers=regularizer, epochs=args.epochs, dropout=args.dropout,
                            batch_size=args.batch_size, optimizer=optimizer,
                            checkpoint_path=args.model_dir, embed_size=args.embed_size,
                            loss=args.loss, metrics=args.metrics, state_sizes=args.state_sizes)

if args.resume is not None:
    model.model = load_model(args.resume)
else:
    model.build()
    model.compile()
model.model.summary()
model.fit()
model.model.save(args.model_dir)