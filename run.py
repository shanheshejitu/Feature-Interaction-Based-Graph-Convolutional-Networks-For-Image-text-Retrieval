from networkgraphwointer import Net
from process_data.data import get_loaders,get_test_loader
from optim import rate,get_optim
from loss import ContrastiveLoss
from evaluation import evalrank
from process_data.data import get_test_loader
from config import Cfgs
import os, torch, datetime, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data

import signal

class Execution:
    def __init__(self, arg):
        self.arg = arg

        if arg.run_mode == 'train':
            print('Loading training set ........')
            self.dataset,self.dataset_eval= get_loaders(arg)   # self.dataset_eval
            #self.dataset_eval = get_loaders(arg)
        self.device = torch.device("cuda:{}".format(arg.device))
        #self.device = torch.device('cuda:1')

        self.loss = ContrastiveLoss(arg, margin=arg.margin, measure='dot', max_violation=True)


    def train(self, dataset, dataset_eval=None):
        # Define model
        net = Net(self.arg)


        net.to(self.device)
        self.loss.to(self.device)
        net.train()

        # Define the multi-gpu training if needed
        # if self.arg.n_gpu > 1:
        #     net = nn.DataParallel(net, device_ids=self.arg.devices_ids)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        # loss_fn = torch.nn.BCELoss(reduction='sum').to(self.device)

        # Load checkpoint if resume training    重新训练，使用和之前相同的数据
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        if self.arg.resume:
            print(' ========== Resume training')
            path = self.arg.model_save_path + 'log_' + str(self.arg.log_id) + '_epoch' + str(self.arg.load_epoch) + '.pth'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            model_dict = torch.load(path)
            print('Finish!')
            net.load_state_dict(model_dict['state_dict'])

            # Load the optimizer paramters
            #start_epoch = model_dict['epoch']+1
            start_epoch = model_dict['epoch']
            lr = model_dict['lr']
            optim = get_optim(net)
            #optim.load_state_dict(model_dict['optimizer'])

        else:
            start_epoch = 0
            lr = self.arg.lr_base
            optim = get_optim(net)


        loss_sum = 0
        #named_params = list(net.named_parameters())   # ！！！
        #grad_norm = np.zeros(len(named_params))       # ！！！


        # Training script
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        for epoch in range(start_epoch, self.arg.max_epoch):

            # Save log information
            logfile = open(
                self.arg.log_path +
                'log_run_' + str(self.arg.log_id) + '.txt',
                'a+'
            )
            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            lr= rate(lr, epoch, self.arg)
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
            for p in optim.param_groups:
                p['lr'] = lr

            time_start = time.time()

            num_i = 0
            # Iteration
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
            for i, (img, txt, index, le) in enumerate(dataset):

                optim.zero_grad()

                img = img.to(self.device)
                txt = txt.to(self.device)
                le = le.to(self.device)
                index = index.to(self.device)


                # for accu_step in range(self.__C.grad_accu_steps):   # 显存不够时，拆分 batch
                #
                #     sub_img_feat_iter = \
                #         img_feat_iter[accu_step * self.__C.sub_batch_size:
                #                       (accu_step + 1) * self.__C.sub_batch_size]
                #     sub_ques_ix_iter = \
                #         ques_ix_iter[accu_step * self.__C.sub_batch_size:
                #                      (accu_step + 1) * self.__C.sub_batch_size]

                scores = net(img, txt, le)

                loss = self.loss(scores)

                loss.backward()

                loss_sum += loss.cpu().data.numpy()

                signal.signal(signal.SIGCHLD, signal.SIG_IGN)
                if self.arg.verbose:
                    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
                    if dataset_eval is not None:
                        mode_str = 'train' + '->' + 'dev'#dev为开发集，用于模型参数调优
                    else:
                        mode_str = 'train' + '->' + 'test'

                    print("\r[log_id %s][epoch %2d][step %4d][%s] loss: %.4f, lr: %.2e" % (
                        self.arg.log_id,
                        epoch + 1,
                        i,
                        mode_str,
                        loss.cpu().data.numpy(),
                        lr
                    ), end='          ')     # 原地打印，不换行

                # Gradient norm clipping
                signal.signal(signal.SIGCHLD, signal.SIG_IGN)
                if self.arg.grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.arg.grad_norm_clip
                    )


                optim.step()
                num_i +=1

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            # print('')
            epoch_finish = epoch + 1

            # Save checkpoint
            state = {
                'state_dict': net.state_dict(),
                #'optimizer': optim.state_dict(),
                'lr': lr,
                'epoch' : epoch_finish
            }
            torch.save(
                state,
                self.arg.model_save_path +
                'log_' + str(self.arg.log_id) +
                '_epoch' + str(epoch_finish) +
                '.pth'
            )

            # Logging ！！！
            logfile = open(
                self.arg.log_path +
                'log_run_' + str(self.arg.log_id) + '.txt',
                'a+'
            )
            logfile.write(
                'epoch = ' + str(epoch_finish) +
                '  loss = ' + str(loss_sum / num_i) +
                '\n' +
                'lr = ' + str(lr) +
                '\n'
            )
            logfile.close()

            # Eval after every epoch
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
            if (dataset_eval is not None) and (epoch >= 15):
                self.eval(
                    self.dataset_eval,
                    state_dict=net.state_dict(),
                )



            loss_sum = 0
            #grad_norm = np.zeros(len(named_params))


    # Evaluation
    def eval(self, data_loader, state_dict=None):

        # Load parameters
        path = self.arg.model_save_path + \
                'log_' + str(self.arg.log_id) + \
                '_epoch' + str(self.arg.load_epoch) + '.pth'
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        if state_dict is None:
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            print('Finish!')


        net = Net(self.arg)

        #net.cuda()
        net.to(self.device)
        net.eval()

        # if self.arg.n_gpu > 1:
        #     net = nn.DataParallel(net, device_ids=self.arg.devices_ids)

        net.load_state_dict(state_dict)

        evalrank(self.arg, net,data_loader, state_dict=state_dict, fold5=False, return_ranks=False)



    def run(self, run_mode):
        if run_mode == 'train':
            # self.empty_log(self.__C.log_id)
            self.train(self.dataset, self.dataset_eval)   # 训练完每一轮进行验证
            #self.train(self.dataset, None)


        elif run_mode == 'test':
            self.eval(get_test_loader(self.arg, 'test'))

        elif run_mode == 'dev':
            self.eval(get_test_loader(self.arg, 'dev'))
        elif run_mode == 'testall':
            self.eval(get_test_loader(self.arg, 'testall'))

        else:
            exit(-1)  # 停止程序




    def empty_log(self, log_id):
        print('Initializing log file ........')
        if os.path.exists(self.arg.log_path +'log_run_' + str(log_id) + '.txt'):
            os.remove(self.arg.log_path +'log_run_' + str(log_id) + '.txt')
        print('Finished!')
        print('')


#if __name__ == '__main__':
#arg='train','/home/cz/czf/retrieval/reasoning/datasets/f30k_precomp',70,0,0,0.3

#my_Execution=Execution(arg=arg)
#my_Execution.run('train')



