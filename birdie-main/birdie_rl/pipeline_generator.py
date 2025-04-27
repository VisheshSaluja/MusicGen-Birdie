# pipeline_generator.py

import os
import threading
import queue
from typing import List
import torch
from datasets import load_dataset
import multiprocessing as mp
import queue as local_queue
import time
import numpy as np
from birdie_rl.pipeline.main_controller import MainController
from birdie_rl.pipeline.worker import Worker

# ✅ Move dataset loading to a CLASS
class MusicDataGenerator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.music_data = np.load(self.data_path, allow_pickle=True)

    def __call__(self):
        while True:
            for sample in self.music_data:
                yield {
                    "input": sample["input"],
                    "label": sample["input"],
                }

def datagen(
    max_batches: int,
    results_q: mp.Queue,
    tasks_q: mp.Queue,
    output_q: queue.Queue,
    sample_q: mp.Queue,
    worker_threads: List[threading.Thread],
    accelerator=None,
    move_to_gpu_fn=None,
    split=None,
):
    print_fn = print if accelerator is None else accelerator.print

    batches_received = 0
    while (max_batches == -1) or (batches_received < max_batches):
        try:
            batch_dict = results_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if batch_dict is None:
            break

        batches_received += 1
        batch = batch_dict["batch_items"]

        if move_to_gpu_fn is not None:
            batch = move_to_gpu_fn(batch)

        output_q.put(batch)

    tasks_q.put(None)
    output_q.put(None)
    os._exit(1)

def samples_to_batch(
    sample_queue: mp.Queue,
    results_queue: mp.Queue,
    batch_size: int,
):
    batch = []
    while True:
        try:
            packed_sample = sample_queue.get(timeout=1.0)
        except Exception:
            continue

        batch.append(packed_sample)

        if len(batch) == batch_size:
            keys = ["input_ids", "label_ids", "segment_ids", "attention_mask"]
            stacked_batch = {
                k: torch.tensor(np.stack([x["packed_data"][k] for x in batch]), dtype=torch.long)
                for k in keys
            }
            batch = []
            results_queue.put({
                "batch_items": stacked_batch,
            })

def pipeline_data_generator(
    max_batches=-1,
    batch_size=8,
    sequence_length=4096,
    num_workers=16,
    objectives_config=None,
    accelerator=None,
    move_to_gpu_fn=None,
    data_generator=None,
    data_generator_fn_kwarg_overrides={},
    infinite_loop=True,
    split=None,
    config={},
):
    assert data_generator is not None, "pipeline_data_generator(): data_generator must not be None."

    if accelerator is None:
        total_workers = num_workers
    else:
        total_workers = num_workers * accelerator.num_processes

    tasks_q = mp.Queue()
    results_q = mp.Queue(8)
    sample_queue = mp.Queue(8)
    output_q = local_queue.Queue(8)

    if objectives_config is None:
        objectives_config = [{"name": "next_token_prediction", "prob": 0.5}]

    worker_threads = []
    our_worker_id_offset = (num_workers * accelerator.process_index) if accelerator else 0

    # ✅ Instead of lambda or local functions
    our_data_generator_class = MusicDataGenerator(config["data_path"])

    for worker_id in range(our_worker_id_offset, our_worker_id_offset + num_workers):
        worker = Worker(
            worker_id=worker_id,
            total_workers=total_workers,
            tasks_queue=tasks_q,
            results_queue=results_q,
            sample_queue=sample_queue,
            sequence_length=sequence_length,
            batch_size=batch_size,
            min_seq_len_for_packing=config.get("min_seq_len_for_packing", 64),
            data_generator=our_data_generator_class,
            infinite_loop=infinite_loop,
            split=split,
            tokenizer=config['tokenizer'],
            text_grabber_fn=config.get("text_grabber_fn", None),
            start_generating_paradigm=config.get("start_generating_paradigm", "\n<|assistant|>\n"),
        )
        worker_thread = mp.Process(target=worker.run)
        worker_threads.append(worker_thread)

    num_bp = 8 if split == 'train' else 1
    for _ in range(num_bp):
        batch_proc = mp.Process(target=samples_to_batch, args=(sample_queue, results_q, batch_size))
        worker_threads.append(batch_proc)

    threading.Thread(target=datagen, kwargs=dict(
        max_batches=max_batches,
        results_q=results_q,
        tasks_q=tasks_q,
        output_q=output_q,
        sample_q=sample_queue,
        worker_threads=worker_threads,
        accelerator=accelerator,
        move_to_gpu_fn=move_to_gpu_fn,
        split=split,
    )).start()

    def _generator():
        while True:
            try:
                yield output_q.get(timeout=1)
            except queue.Empty:
                continue

    generator = _generator()

    main_ctrl = MainController(
        tasks_queue=tasks_q,
        results_queue=results_q,
        objectives_config=objectives_config,
        num_workers=num_workers,
        max_batches=max_batches,
    )

    main_ctrl.run()

    for worker_thread in worker_threads:
        worker_thread.start()

    return (main_ctrl, generator)
