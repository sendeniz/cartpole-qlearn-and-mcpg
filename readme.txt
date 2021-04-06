Tabular Q-learning:
    
    TabularQlearning_FindBins.py runs a for-loop over all tried values of numbers of bins for tabular Q-learning: 1 through 6 for all parameters, except pole angle, which was (later) increased to go up to 10. This files combines all.

    TabularQlearning_optimal.py runs the tabular Q-learning architecture for number of bins [1,1,6,3].
    
    TabularQlearning_learningschedules.py runs four different models, three constant exploration parameters, and one exploration parameter that follows a learning schedule.



Deep Q-learning:
    
    Files for the DQN require the keras tensorflow package and !pip install keras-layer-normalization should be run to make sure Layer Normalisation is installed. 
    
    DeepQlearning_Experiments.py runs models to fine-tune the neural network hyperparameters. Three models for the number of hidden layers, four models to determine which activation function to use and three models regarding model regularization.



Monte Carlo Policy gradient:

    Files for MCPG require the PyTorch package.
    
    MonteCarloPolicygradient_Hyperparameters.py runs a for-loop over all tried values of hyperparameters for MCPG: learning rate, discount factor, hidden size, number of layers. It outputs a figure of the training sequence: total reward as a function of episode.
    
    MonteCarloPolicygradient_Freeze.py runs an algorithm where, once 10 successive perfect scores are attained, the model freezes and the rest of the episodes are finished.
    
    

Learning schedule:
    LearningSchedule.py creates the plot for parameter values as a function of episode.
