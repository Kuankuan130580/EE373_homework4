import numpy as np

class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def fit(self, env, steps=100):
        """
        Trains the MultiArmedBandit on an OpenAI Gymnasium environment.

        See page 32 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. For the step size, use
        1/N, where N is the number of times the current action has been
        performed. (This is the version of Bandits we saw in lecture before
        we introduced alpha). Use an epsilon-greedy approach to pick actions.

        See (https://www.gymlibrary.dev/) for examples of how to use the OpenAI
        Gymnasium Environment interface.

        In every step of the fit() function, you should sample
            two random numbers using functions from `src.random`.
            1.  First, use either `src.random.rand()` or `src.random.uniform()`
                to decide whether to explore or exploit.
            2. Then, use `src.random.choice` or `src.random.randint` to decide
                which action to choose. Even when exploiting, you should make a
                call to `src.random` to break (possible) ties.

        Please don't use `np.random` functions; use the ones from `src.random`!
        Please do not use `env.action_space.sample()`!

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "terminated or truncated" returned
            from env.step() is True.
          - When choosing to exploit the best action rather than exploring,
            do not use np.argmax: it will deterministically break ties by
            choosing the lowest index of among the tied values. Instead,
            please *randomly choose* one of those tied-for-the-largest values.
          - MultiArmedBandit treats all environment states the same. However,
            in order to have the same API as agents that model state, you must
            explicitly return the state-action-values Q(s, a). To do so, just
            copy the action values learned by MultiArmedBandit S times, where
            S is the number of states.

        Arguments:
          env - (Env) An OpenAI Gymnasium environment with discrete actions and
            observations. See the OpenAI Gymnasium documentation for example use
            cases (https://www.gymlibrary.dev/api/core/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length `num_bins`.
            Let s = int(np.ceil(steps / `num_bins`)), then rewards[0] should
            contain the average reward over the first s steps, rewards[1]
            should contain the average reward over the next s steps, etc.
        """
        num_bins = 2
        # set up Q function, rewards
        n_actions, n_states = env.action_space.n, env.observation_space.n
        self.Q = np.zeros(n_actions)
        self.N = np.zeros(n_actions)
        avg_rewards = np.zeros([num_bins])
        all_rewards = []

        # reset environment before your first action
        env.reset()
        total_sum=1
        for i in range(steps):
          explore=1 if np.random.uniform(0,1)<self.epsilon else 0
          if explore:
            A=np.random.randint(n_actions)
          else:
            q_max=np.max(self.Q)
            index=np.where(self.Q==q_max)[0]
            A=np.random.choice(index)
          observation, reward, terminated, truncated, info = env.step(A)
          total_sum *= (1+reward)
          all_rewards.append(reward)
          self.N[A]+=1
          self.Q[A]+=1/(self.N[A])*(reward-self.Q[A])  
          if terminated or truncated:
              env.reset()
        for k in range(num_bins):
           s=int(np.ceil(steps/num_bins))
           avg_rewards[k]=np.mean(all_rewards[k*s:(k+1)*s])
        state_action_values=np.tile(self.Q,(n_states,1))
        return  state_action_values,avg_rewards,total_sum

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. During prediction, you
            should only "exploit."
          - You should use a loop to predict over each step in an episode until
            it terminates by returning `terminated or truncated=True`.
          - When choosing to exploit the best action, do not use np.argmax: it
            will deterministically break ties by choosing the lowest index of
            among the tied values. Instead, please *randomly choose* one of
            those tied-for-the-largest values.

        Arguments:
          env - (Env) An OpenAI Gymnasium environment with discrete actions and
            observations. See the OpenAI Gymnasium documentation for example use
            cases (https://www.gymlibrary.dev/api/core/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        # reset environment before your first action
        env.reset()
        
        state=[]
        actions=[]
        rewards=[]
        while(1):
           q_max=np.max(state_action_values[0])
           index=np.where(state_action_values[0]==q_max)[0]
           A=np.random.choice(index)
           observation, reward, terminated, truncated, info = env.step(A)
           actions.append(A)
           rewards.append(reward)
           if terminated or truncated:
            state.append(actions)
            break
        return state,actions,rewards  