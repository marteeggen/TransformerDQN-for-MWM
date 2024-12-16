from typing import Any
from dataclasses import dataclass
from random import seed, randint
from collections import deque

seed(1337)

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    def __init__(self, buffer_size=50000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def insert(self, sarsd): 
        self.buffer.append(sarsd)
    
    def sample(self, num_samples, sequence_length):
        assert num_samples < min(len(self.buffer), self.buffer_size)

        max_offset_index = len(self.buffer) - sequence_length
        offsets = [randint(0, max_offset_index) for _ in range(num_samples)]
        sequences = [list(self.buffer)[offset:offset + sequence_length] for offset in offsets]
        return sequences
