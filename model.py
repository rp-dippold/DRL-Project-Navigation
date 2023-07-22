import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    """
    This class represents the policy model of an actor.

    Class attributes
    ----------------
    ln_in : nn.Linear
        Linear input layer
    hidden : nn.ModuleList
        List of linear hidden layers
    ln_out : nn.Linear
        Linear output layer

    Methods
    -------
    forward(state)
        Returns the action suggested by the policy model for a state.
        In fact, the action proposed by the policy is the tensor element
        with the highest value to be selected by the calling function.
    """

    def __init__(self, state_size, action_size, hidden_sizes):
        """Initialize a deep Q network.
        
        Parameters
        ----------
        state_size : int
            Dimension of each state
        action_size : int
            Dimension of each action
        hidden_size : list
            List of the number of neurons in each hidden layer.
            Has to contain at least one element.
        """
        super(DQNetwork, self).__init__()
        # linear input layer
        self.ln_in = nn.Linear(state_size, hidden_sizes[0])
        # stack of linear hidden layers
        self.hidden = nn.ModuleList()
        for idx in range(len(hidden_sizes)-1):            
            self.hidden.append(nn.Linear(hidden_sizes[idx], 
                                         hidden_sizes[idx+1]))
        # linear output layer
        self.ln_out = nn.Linear(hidden_sizes[-1], action_size)

    def forward(self, state):
        """Returns the policy's action for a state provided as argument.

        Argument `state` is the state of the environment for which
        an action should be returned according to the policy represented
        by the neural network.

        Parameters
        ----------
        state : array-like (37 elements)
            The state (observation) of the game.

        Returns
        -------
            The next action (tensor) according to the policy.
        """
        x = F.relu(self.ln_in(state))
        for hd in self.hidden:
            x = F.relu(hd(x))
        return self.ln_out(x)
