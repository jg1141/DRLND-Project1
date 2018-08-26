# DRLND-Project1
DRLND Project 1 - Navigation (Banana environment)

---

The notebook and Python files in this repository present a solution using the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).



## Project Details

The state space for this project is a continuous 37-dimension vector representing the position and ray angles of bananas relative to the agent's forward direction. The action space consists of the four possibilities: 

- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

An episode score is the number of YELLOW bananas collected minus the number of BLUE bananas collected.

The environment is considered solved when the agent has an average score of 13 or higher in 100 episodes.

## Getting Started

#### Install the Anaconda distribution of Python 3

[Anaconda Python](https://www.anaconda.com/download/#macos) installation.

#### Obtain Unity ML-Agents

Install [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

## Instructions

#### Start Jupyter and open the DRLND Project 1 notebook

```
> jupyter notebook
```

[DRLND Project 1 - Navigation (Banana environment).ipynb](http://localhost:8888/notebooks/DRLND%20Project%201%20-%20Navigation%20(Banana%20environment)/DRLND%20Project%201%20-%20Navigation%20(Banana%20environment).ipynb) - link for use on your localhost:8888 Jupyter instance

Follow instructions there to obtain the **Unity3D Banana environment** from Udacity.

Select **Cell > Run All**.

#### Additional Resources

Need more links? Visit [http://bit.ly/drlndlinks](http://bit.ly/drlndlinks) to learn much more about DRLND and Reinforcement Learning.

![Image of Unity3D Banana environment](https://github.com/jg1141/DRLND-Project1/blob/master/Unity3D%20Banana%20environment.png)
