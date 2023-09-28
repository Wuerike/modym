# Modym 

Base environment to apply reinforcement learning in Modelica models through Gymnasium API.

This project is **HIGHLY** inspired by [modelicagym](https://github.com/ucuapps/modelicagym), without it Modym wouldn't exist!

Despite the similarities, there are some reasons that justify Modym existence:
- Updated dependencies 
- Ready to use docker environment 

## ModymEnv

ModymEnv is the base environment class, which implements the Gymnasium API and executes simulation steps in Modelica compiled models (FMU files). It's a merge of many base classes from the modelicagym project, such as `ModelicaBaseEnv`, `ModelicaCSEnv` and `ModelicaMEEnv`.

It has two main responsibilities:
1. Reset the simulation to a certain state, which can be changed at each step through `options` parameter or can be always the `model_parameters` config
2. Apply actions into Modelica models, simulating it during the configured time step and returning the result of this simulation

Beyond this, it holds some methods you'd have to implement in your environment, sush as `_get_action_space`, `_get_observation_space` and others.

### PyFMI

The [PyFMI](https://github.com/modelon-community/PyFMI) package is used to interact with the FMU files.

## Docker Environment

The image `wuerike/modym:v1`, built from the `Dockerfile`, provides a ready to run environment with all dependencies.

### Run With DevContainer - [Best Way]

If you use Visual Studio Code, you can run with the [DevContainer Extension](https://code.visualstudio.com/docs/devcontainers/containers).

The `.devcontainer/devcontainer.json` file describes the environment to be created.

By installing the extension and executing the command `Dev Containers: Reopen in Container`, it will reopen your vscode session inside the docker image.

Your workspace will be shared as a volume, so all change will be applied to both local and container environment.

### Run With Docker Compose

By executing `docker compose up` your workspace will be shared as a volume, but a vscode session inside the container will not be created.

Then you should attach the container terminal to yours and run all the commands through it.

### Sharing Display

As some dependencies may require window creation, as matplotlib, X11 folder its being shared as volume, then windows can be opened from inside the container.

AS it was developed on Ubuntu, if you require display usage and you are in other OS, changes may be necessary.

You may also need run `xhost +` in your machine to your display be available to the container.

## CartPole example

`CartPoleEnv` implements `ModymEnv` and `CartPoleLearner` solve it with Q-learning algorithm.

It uses the FMU files available on [modelicagym](https://github.com/ucuapps/modelicagym).

### Setup

The `config.json` has `CartPoleEnv`'s and `ModymEnv`'s required configurations.

Son the `CartPoleLearner`'s parameters will be also added to it, then all the simulation setup will be centralized.

### How To Run

#### Random Solver

Takes random actions:
```shell
python main.py -r
```

#### Train a Q-Learning Policy

Train and test an agent:
```shell
python main.py
```

At the end, creates a file on `/policies` so you can reexecute the trained agent.

#### Run an Existing Policy

Execute policies from `/policies` folder:
```shell
python main.py -p great_policy.csv
```

## Creating a New Environment

There's no currently plan of sharing Modym as a pip package.

The value it offers is its ready to run environment, so go ahead, fork this repository, create a new folder to your brand new environment and develop it.

And if you do so, please don't hesitate on sharing it with me, I'll be glad to see your work!

## Credits

- [Oleh Lukianykhin](https://github.com/OlehLuk): For being the main author of [modelicagym](https://github.com/ucuapps/modelicagym) which is the base of this project
- [Yangyang Fu](https://github.com/YangyangFu): For being the main author of [FMU-DRL-DOCKER](https://github.com/BE-HVACR/FMU-DRL-DOCKER) which gave me the inspiration of running reinforcement learning inside of a docker container

## License

This project is licensed under the GPLv3 - see the [LICENSE.md](LICENSE.md) file for details.
