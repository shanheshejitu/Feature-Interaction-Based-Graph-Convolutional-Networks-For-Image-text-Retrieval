from types import MethodType


class PATH:
    def __init__(self):

        # dataset root path
        self.dataset_path = '/home/gaofl/datasets/f30k_precomp'

        # bottom up features root path
        self.feature_path = '/home/gaofl/datasets/coco_precomp'
        self.pretrained_emb_path = '/home/gaofl/datasets/glove6b_init_300d.npy'

        self.init_path()


    def init_path(self):

        self.img_feat_path = {
            'train': self.feature_path + 'train2014/',
            'dev': self.feature_path + 'val2014/',
            'test': self.feature_path + 'test2015/',
            'testall': 1
        }


        self.result_path = 1
        self.model_save_path = "/home/gaofl/GCNw-r inter para 3 - 147/save/"
        self.log_path = "/home/gaofl/GCNw-r inter para 3 - 147/save/"
        self.eval_path = 1


class Cfgs(PATH):
    def __init__(self):
        super(Cfgs, self).__init__()





        self.load_epoch = 0

        self.load_path = None     # load pre-trained model

        # ------------------------------
        # ---- Data Provider Params ----
        # ------------------------------

        # {'train', 'val', 'test'}
        self.run_mode = 'train'

        # Set True to evaluate offline
        self.eval_every_epoch = True

        # Set True to use pretrained word embedding
        # (GloVe: spaCy https://spacy.io/)
        self.use_glove = True

        # Word embedding matrix size
        self.word_embed_size = 300

        self.num_embeddings = 400002

        # Max length of extracted faster-rcnn 2048D features
        # (bottom-up and Top-down: https://github.com/peteanderson80/bottom-up-attention)
        self.img_feat_pad_size = 100

        # Faster-rcnn 2048D features
        self.img_feat_size = 2048

        # Default training batch size: 64
        self.batch_size = 56   # biaozun60

        self.eval_batch_size = int(self.batch_size / 2)

        # Multi-thread I/O
        self.num_workers = 0

        # Use pin memory
        # (Warning: pin memory can accelerate GPU loading but may
        # increase the CPU memory usage when NUM_WORKS is large)
        self.pin_mem = False

        # Gradient accumulate can split batch to reduce gpu memory usage
        # (Warning: BATCH_SIZE should be divided by GRAD_ACCU_STEPS)
        self.grad_accu_steps = 1

        # ------------------------
        # ---- Network Params ----
        # ------------------------

        self.n_res_blocks = 1

        self.max_length = 22

        self.HIDDEN_SIZE = 512

        self.DROPOUT_R = .1

        self.dropout_r = .1

        self.MULTI_HEAD = 4

        self.HIDDEN_SIZE_HEAD = 128


        self.GRU_hidden = 512

        self.FF_SIZE = 1024

        # --------------------------
        # ---- Optimizer Params ----
        # --------------------------

        # The base learning rate    0.0002
        self.margin = 0.335  # 0.2

        self.lr_base = 0.000375


        # Learning rate decay ratio
        self.lr_decay_r = 0.5
        #self.lr_decay_r1=1.3
        self.hyper = 1.5

        # Learning rate decay at {x, y, z...}
        self.lr_decay_list = [7,10,14,18,22,26,30,34,38]
        #self.lr_decay_list = [8,11,15,19,23,27,30,34,38]

        # Max training epoch
        self.max_epoch = 41

        # Gradient clip
        # (default: -1 means not using)
        self.grad_norm_clip = 0.25     # 0.25

        # Adam optimizer betas and eps
        self.opt_betas = (0.9, 0.999)     # (0.9, 0.98)
        self.opt_eps = 1e-8               # 1e-9


    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict



    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])





    def __str__(self):
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                print('{ %-17s }->' % attr, getattr(self, attr))

        return ''

