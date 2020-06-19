import random
import numpy as np
import glob
import joblib


class ReplayMemory:
    def __init__(self, capacity, split=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

        self.split = split
        self.saved_num = -1
        self.filelist = [0]
        self.num = 0
        assert capacity % split == 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        try:
            self.num = self.num + 1
            self.position = self.num % self.capacity
        except:
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def save(self, filename_body, filename_ext, verbose=True):
        if verbose:
            print("=> save replay memory '{}'".format(filename_body))

        saved_num = self.saved_num
        self.saved_num = self.num-1

        start_index = (saved_num+1) // self.split
        end_index = (self.num-1) // self.split
        max_index = self.capacity // self.split

        for index in range(start_index, end_index+1):
            _index = index % max_index
            filename = '{}_{}.{}'.format(filename_body, _index, filename_ext)
            start_buffer = _index*self.split
            end_buffer = min((_index+1)*self.split, len(self.buffer))
            state = {
                    'index': index,
                    'start_buffer': start_buffer,
                    'end_buffer': end_buffer,
                    'buffer': self.buffer[start_buffer:end_buffer],
                    'position': self.position,
                    'num': self.num,
                    'saved_num': self.saved_num,
                    'capacity': self.capacity,
                    'split': self.split,
                    }
            joblib.dump(state, filename)
            if verbose:
                #print(_index)
                print("=>     save replay memory '{}'".format(filename))
        if verbose:
            print('=> done')

    def load(self, filename_body, filename_ext):
        max_index = self.capacity // self.split
        states = []
        cur_index = 0
        cur_num = 0
        cur_saved_num = 0
        for _index in range(0, max_index):
            filename = '{}_{}.{}'.format(filename_body, _index, filename_ext)
            try:
                state = joblib.load(filename)
                index = state['index']
                num = state['num']
                saved_num = state['saved_num']
                self.split = state['split']
                self.capacity = state['capacity']

                #_index = index % max_index
                #end_buffer = state['end_buffer']
                #end_buffer = min((_index+1)*self.split, len(self.buffer))
                cur_index = max(cur_index, index)
                cur_num = max(cur_num, num)
                cur_saved_num = max(cur_saved_num, saved_num)
                states += [state]
            except:
                pass

        # init
        self.buffer = [None]*min(cur_num, self.capacity)
        self.num = cur_num
        self.saved_num = cur_saved_num
        self.position = self.num % self.capacity

        # update
        for state in states:
            _buffer = state['buffer']
            start_buffer = state['start_buffer']
            end_buffer = state['end_buffer']
            self.buffer[start_buffer:end_buffer] = _buffer

        # end
        assert None not in self.buffer
        #print(self.num, self.saved_num, self.position)

    def __len__(self):
        return len(self.buffer)
