import numpy as np
import copy
import heapq
import os

# 🔥 NEW: Utility function to detect if it's a music dataset
def is_music_dataset(file_path):
    return file_path.endswith(".npy") and "music" in os.path.basename(file_path).lower()

def debug_alignments(current_dict, sub_idx=0):
    for idx in range(len(current_dict["input_ids"][sub_idx])):
        print(
            f"idx: {idx}, "
            f"input_ids: {current_dict['input_ids'][sub_idx][idx]}, "
            f"label_ids: {current_dict['label_ids'][sub_idx][idx]}, "
            f"segment_ids: {current_dict['segment_ids'][sub_idx][idx]}, "
            f"attention_mask: {current_dict['attention_mask'][sub_idx][idx]}"
        )

class Packer:
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = copy.deepcopy(config)
        del config

        self.minimum_sequence_length = self.config.get("minimum_sequence_length", 64)
        self.sequence_length = int(self.config.get("sequence_length", 1024))
        self.start_generating_paradigm = self.config.get("start_generating_paradigm", "\n<|assistant|>\n")
        
        # 🔥 Music data special handling
        self.music_mode = self.config.get("music_mode", False)
        if not self.music_mode:
            self.tokenized_start_generating_paradigm = self.config.get("tokenizer", None).encode(self.start_generating_paradigm)
        else:
            self.tokenized_start_generating_paradigm = np.zeros((1,))  # Dummy for music

        self.reset(self.sequence_length)
        assert self.sequence_length >= self.minimum_sequence_length

    def reset(self, sequence_length=None):
        target_sequence_length = sequence_length or self.sequence_length

        self.current_dict = {
            "input_ids": np.zeros((target_sequence_length,), dtype=np.float32 if self.music_mode else np.int32),
            "attention_mask": np.zeros((target_sequence_length,), dtype=np.int32),
            "label_ids": np.zeros((target_sequence_length,), dtype=np.float32 if self.music_mode else np.int32) - 100,
            "segment_ids": np.zeros((target_sequence_length,), dtype=np.int32),
        }
        self.remaining_space = target_sequence_length
        self.sequence_length = target_sequence_length
        self.data_index = 0
        self.segment_counter = 0

        return self

    def get_remaining_space(self):
        paradigm_len = len(self.tokenized_start_generating_paradigm)
        return max(0, self.remaining_space - paradigm_len)

    def is_ready(self):
        return self.remaining_space <= self.minimum_sequence_length

    def can_accept(self, input_ids, label_ids):
        paradigm_len = len(self.tokenized_start_generating_paradigm)
        total_length_to_add = len(input_ids) + paradigm_len + len(label_ids) - 1
        return total_length_to_add <= self.remaining_space

    def add(self, input_ids, label_ids, loss_mask=None):
        paradigm_len = len(self.tokenized_start_generating_paradigm)
        total_length_to_add = len(input_ids) + paradigm_len + len(label_ids) - 1

        if not self.can_accept(input_ids, label_ids):
            raise ValueError(
                f"Insufficient space to add {total_length_to_add:,} tokens.  "
                f"remaining_space: {self.remaining_space:,}"
            )
        
        # 🔥 If in text mode, append paradigm tokens
        if not self.music_mode:
            input_ids = np.concatenate([input_ids, self.tokenized_start_generating_paradigm])

        self.segment_counter += 1
        
        input_start = self.data_index
        input_end = input_start + len(input_ids)
        label_start = input_end - 1
        label_end = label_start + len(label_ids)

        input_teacher_forcing_start = input_end
        input_teacher_forcing_end = label_end

        assert (input_teacher_forcing_end - input_start) == total_length_to_add

        self.current_dict["input_ids"][input_start:input_end] = input_ids
        self.current_dict["attention_mask"][input_start:input_end] = 1
        self.current_dict["input_ids"][input_teacher_forcing_start:input_teacher_forcing_end] = label_ids[:-1]
        self.current_dict["label_ids"][label_start:label_end] = label_ids
        self.current_dict["segment_ids"][input_start:input_teacher_forcing_end] = self.segment_counter

        self.data_index += total_length_to_add
        self.remaining_space -= total_length_to_add

        return self.is_ready()

class Batcher:
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = copy.deepcopy(config)
        del config

        self.batch_size = self.config.get("batch_size", 8)
        self.minimum_sequence_length = self.config.get("minimum_sequence_length", 64)
        self.sequence_length = int(self.config.get("sequence_length", 1024))

        self.music_mode = self.config.get("music_mode", False)

        self.packers = []
        self.pq = []
        self.reset(self.sequence_length, self.batch_size)

    def reset(self, sequence_length=None, batch_size=None):
        target_sequence_length = sequence_length or self.sequence_length
        target_batch_size = batch_size or self.batch_size

        self.sequence_length = target_sequence_length
        self.batch_size = target_batch_size

        self.packers = []
        self.pq = []

        for i in range(self.batch_size):
            packer_config = copy.deepcopy(self.config)
            packer_config["music_mode"] = self.music_mode
            packer = Packer(config=packer_config)
            self.packers.append(packer)
            heapq.heappush(self.pq, (packer.get_remaining_space(), i, packer))

        return self

    def get_remaining_space(self, max_or_min="max"):
        if max_or_min == "max":
            return max(entry[2].get_remaining_space() for entry in self.pq)
        else:
            return min(entry[2].get_remaining_space() for entry in self.pq)

    def is_ready(self):
        if all(entry[2].is_ready() for entry in self.pq):
            return "ready"
        else:
            return "not ready"

    def can_accept(self, input_ids, label_ids):
        return any(entry[2].can_accept(input_ids, label_ids) for entry in self.pq)

    def add(self, input_ids, label_ids, loss_mask=None, force_finish_pack=False):
        buffer = []
        packer_found = False

        while self.pq:
            remaining_space, idx, packer = heapq.heappop(self.pq)

            if packer.can_accept(input_ids, label_ids):
                packer.add(input_ids, label_ids, loss_mask=loss_mask)
                new_remaining_space = 0 if force_finish_pack else packer.get_remaining_space()
                heapq.heappush(self.pq, (new_remaining_space, idx, packer))
                packer_found = True
                break
            else:
                buffer.append((remaining_space, idx, packer))

        while buffer:
            heapq.heappush(self.pq, buffer.pop())

        if not packer_found:
            return "full"

        return self.is_ready()

    def pop(self, peek=False):
        sorted_packers = sorted(self.pq, key=lambda x: x[1])
        keys_to_stack = sorted_packers[0][2].current_dict.keys()
        stacked_dict = {
            key: np.stack([p[2].current_dict[key] for p in sorted_packers])
            for key in keys_to_stack
        }

        if not peek:
            self.reset()

        return stacked_dict

    def get_sample_count(self):
        running_total = 0
        for entry in self.pq:
            running_total += entry[2].segment_counter
        return running_total
