import time
from utils import *
import torch
import torch.multiprocessing as mp
from collections import Counter

ctx = mp.get_context("spawn")

class MultiProcessWorker(ctx.Process):
    # TODO: Make environment init threadsafe
    def __init__(self, id, trainer_maker, comm, main_args, seed, *args, **kwargs):
        self.id = id
        self.seed = seed
        super(MultiProcessWorker, self).__init__()
        self.trainer = trainer_maker()
        self.comm = comm
        self.args = main_args

    def calcu_entropy(self, comm):
        if self.args.test_quant == False:
            comm = 0.5*comm*self.args.quant_levels
            comm = np.rint(comm)
            comm = 2*comm/self.args.quant_levels
        symbol_list = Counter(comm.flatten())
        symbol_counts = np.array(list(symbol_list.values()))
        symbol_p = symbol_counts/np.sum(symbol_counts)
        entropy = - np.sum(symbol_p*np.log(symbol_p))
        return entropy


    def run(self):
        torch.manual_seed(self.seed + self.id + 1)
        np.random.seed(self.seed + self.id + 1)

        while True:
            task = self.comm.recv()
            if type(task) == list:
                task, epoch = task

            if task == 'quit':
                return
            elif task == 'run_batch':
                batch, stat, comm_info = self.trainer.run_batch(epoch)
                if self.args.calcu_entropy:
                    #calculate entropy here
                    comm_np = comm_info.detach().cpu().numpy()    
                    comm_np_list = np.hsplit(comm_np,self.args.msg_size) #split matrix for parallelization
                    entropy_set = map(self.calcu_entropy, comm_np_list)
                    final_entropy = sum(entropy_set)
                    entro_stat = {'comm_entropy':final_entropy}
                    merge_stat(entro_stat, stat)
                self.trainer.optimizer.zero_grad()
                s = self.trainer.compute_grad(comm_info, batch)
                merge_stat(s, stat)
                self.comm.send(stat)
            elif task == 'test_batch':
                comm_stat, steps_taken, success_times = self.trainer.test(epoch)
                if self.args.calcu_entropy:
                    #calculate entropy here
                    comm_np = comm_stat.detach().cpu().numpy()    
                    comm_np_list = np.hsplit(comm_np,self.args.msg_size) #split matrix for parallelization
                    entropy_set = map(self.calcu_entropy, comm_np_list)
                    final_entropy = sum(entropy_set)    
                else: 
                    final_entropy = 0    
                self.comm.send((final_entropy, steps_taken, success_times))
            elif task == 'send_grads':
                grads = []
                for p in self.trainer.params:
                    if p._grad is not None:
                        grads.append(p._grad.data)

                self.comm.send(grads)


class MultiProcessTrainer(object):
    def __init__(self, args, trainer_maker):
        self.comms = []
        self.trainer = trainer_maker()
        # itself will do the same job as workers
        self.nworkers = args.nprocesses - 1
        for i in range(self.nworkers):
            comm, comm_remote = ctx.Pipe()
            self.comms.append(comm)
            worker = MultiProcessWorker(i, trainer_maker, comm_remote, args, seed=args.seed)
            worker.start()
        self.grads = None
        self.worker_grads = None
        self.is_random = args.random
        self.args = args

    def quit(self):
        for comm in self.comms:
            comm.send('quit')

    def obtain_grad_pointers(self):
        # only need perform this once
        if self.grads is None:
            self.grads = []
            for p in self.trainer.params:
                if p._grad is not None:
                    self.grads.append(p._grad.data)

        if self.worker_grads is None:
            self.worker_grads = []
            for comm in self.comms:
                comm.send('send_grads')
                self.worker_grads.append(comm.recv())

    def calcu_entropy(self, comm):
        if self.args.test_quant == False:
            comm = 0.5*comm*self.args.quant_levels
            comm = np.rint(comm)
            comm = 2*comm/self.args.quant_levels
        symbol_list = Counter(comm.flatten())
        symbol_counts = np.array(list(symbol_list.values()))
        symbol_p = symbol_counts/np.sum(symbol_counts)
        entropy = - np.sum(symbol_p*np.log(symbol_p))
        return entropy

    def test_batch(self,times):
        for comm in self.comms:
            comm.send(['test_batch', times])        

        # run its own trainer
        comm_stat_acc, steps_taken_acc, success_times_acc = self.trainer.test(times)
        if self.args.calcu_entropy:
            #calculate entropy here
            comm_np = comm_stat_acc.detach().cpu().numpy()    
            comm_np_list = np.hsplit(comm_np,self.args.msg_size) #split matrix for parallelization
            entropy_set = map(self.calcu_entropy, comm_np_list)
            final_entropy = sum(entropy_set)  
        for comm in self.comms:
            entropy, steps_taken, success_times = comm.recv()
            steps_taken_acc =  np.concatenate((steps_taken_acc,steps_taken), axis=0)
            final_entropy =  np.concatenate((final_entropy,entropy), axis=0)
            success_times_acc =  np.concatenate((success_times_acc,success_times), axis=0)
        
        print('success: ', np.mean(final_entropy),' std: ', np.std(final_entropy) )
        print('success: ', np.mean(success_times_acc),' std: ', np.std(success_times_acc) )
        print('steps: ', np.mean(steps_taken_acc),' std: ', np.std(steps_taken_acc))
           
        return

    def train_batch(self, epoch):
        # run workers in parallel
        for comm in self.comms:
            comm.send(['run_batch', epoch])

        # run its own trainer
        batch, stat, comm_info_acc = self.trainer.run_batch(epoch)
        self.trainer.optimizer.zero_grad()
        s = self.trainer.compute_grad(comm_info_acc, batch)
        merge_stat(s, stat)

        if self.args.calcu_entropy:
            #calculate entropy here
            comm_np = comm_info_acc.detach().cpu().numpy()    
            comm_np_list = np.hsplit(comm_np,self.args.msg_size) #split matrix for parallelization
            entropy_set = map(self.calcu_entropy, comm_np_list)
            final_entropy = sum(entropy_set)   
            entro_stat = {'comm_entropy':final_entropy}
            merge_stat(entro_stat, stat)
        
        # check if workers are finished
        for comm in self.comms:
            s= comm.recv()
            merge_stat(s, stat)

        # add gradients of workers
        self.obtain_grad_pointers()
        for i in range(len(self.grads)):
            for g in self.worker_grads:
                self.grads[i] += g[i]
            self.grads[i] /= stat['num_steps']

        self.trainer.optimizer.step()
        return stat

    def state_dict(self):
        return self.trainer.state_dict()

    def load_state_dict(self, state):
        self.trainer.load_state_dict(state)
