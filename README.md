This is the repository for RCDT(Return-Cost Regularized Constrained Decision Transformer), an offline safe RL algorithm.

## Structure
The structure of this repo is as follows:
```
├── examples
│   ├── configs  # the training configs of each algorithm
│   ├── eval     # the evaluation escipts
│   ├── train    # the training scipts
├── osrl
│   ├── algorithms  # offline safe RL algorithms
│   ├── common      # base networks and utils
```



## Installation

OSRL is currently hosted on [PyPI](https://pypi.org/project/osrl-lib), you can simply install it by:

```bash
pip install osrl-lib
```


Install the `OApackage`:
```bash
pip install OApackage==2.7.6
```

## How to use OSRL

The example usage are in the `examples` folder, where you can find the training and evaluation scripts for all the algorithms. 
All the parameters and their default configs for each algorithm are available in the `examples/configs` folder. 
OSRL uses the `WandbLogger` in [FSRL](https://github.com/liuzuxin/FSRL) and [Pyrallis](https://github.com/eladrich/pyrallis) configuration system. The offline dataset and offline environments are provided in [DSRL](https://github.com/liuzuxin/DSRL), so make sure you install both of them first.

### Training

```shell
python examples/train/train_c2dt.py --task OfflineCarCircle-v0 --param1 args1 ...
```
By default, the config file and the logs during training will be written to `logs\` folder and the training plots can be viewed online using Wandb.

You can also launch a sequence of experiments or in parallel via the EasyRunner package, see `examples/train_all_tasks.py` for details.

### Evaluation
To evaluate a trained agent, for example, a RCDT agent, simply run
```shell
python examples/eval/eval_c2dt.py --path path_to_model --eval_episodes 20
```
It will load config file from `path_to_model/config.yaml` and model file from `path_to_model/checkpoints/model.pt`, run 20 episodes, and print the average normalized reward and cost.

## Acknowledgement

The framework design and most baseline implementations of OSRL are heavily inspired by the [CORL](https://github.com/corl-team/CORL) project, which is a great library for offline RL, and the [cleanrl](https://github.com/vwxyzjn/cleanrl) project, which targets online RL. So do check them out if you are interested!


## Contributing

If you have any suggestions or find any bugs, please feel free to submit an issue or a pull request. We welcome contributions from the community! 
