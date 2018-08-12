
---
### Checkpoints
Users can specify the checkpoint strategies for an experiment (a `run`).

#### Path and file names
Given a `run_id`, the checkpoint directory for a `run` is set to
```python
ckpt_run_dir = "{checkpoint_root}/{run_id}_{dataset}_{model}_{optimizer}/"
```
Note that the `dataset`, `model` and `optimizer` are used to improve readability. The configuration file containing all of the parameters and related settings are stored in `ckpt_run_dir/config.json`.
The `run_id` is given by the mlbench-master to each experiment. It can be an integer or other more readable index.

During a `run`, workers periodically checkpoint and save them in `ckpt_run_dir/`. 
The name of such checkpoints follw pattern
```python
# Checkpoint generated at epoch_id batch_id at rank process.
ckpt_id = "{epoch}_{rank}"
```
Since `ckpt_run_dir` could be the same directory, `rank` is appended to `ckpt_id` to avoid collision. 

Inside one `ckpt_id`, we saves 

- the model weights of `rank`;
- the validation score of the model weights;
- the metrics value of the model weights;
- other information with regards to run time;
- include the dataloader they are using.

#### Restore
*Now we only consider the synchronized case. The asynchronized case is left to be done.*

To restore an experiment, a worker needs to know the `run_id`, `epoch`. By default, latest `run_id`, `epoch` which shared by all workers are restored.

When resuming a checkpoint, program first load the configuration file in `ckpt_run_dir` and then restore all of the weights, scores, metrics, etc in `ckpt_id`. The running time clock continues after all workers reach the same barrier settled at the end of `restore`.


#### TODO
- make sure that the random generator is the same as one without checkpoint at that epoch/batch.
- checkpointing and resuming for the asynchronized learning.