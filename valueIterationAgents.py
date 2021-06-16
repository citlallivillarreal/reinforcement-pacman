# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
    
        new_values = util.Counter()
        for i in range(self.iterations):
          for state in self.mdp.getStates():
            action = self.computeActionFromValues(state)
            if action != None: 
              curr_val = self.computeQValueFromValues(state, action)
            else: 
              curr_val = 0
            new_values[state] = curr_val
          self.values = new_values
          new_values = new_values.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        nextS_probs_lst = self.mdp.getTransitionStatesAndProbs(state, action)
        nextS = [j[0] for j in nextS_probs_lst]
        probs = [k[1] for k in nextS_probs_lst]
        q_value = 0
        for nextstate in range(len(nextS)): 
          prob = probs[nextstate]
          successor = nextS[nextstate]
          reward = self.mdp.getReward(state, action, successor)
          value = self.getValue(successor)
          q_v = prob*(reward+self.discount*value)
          q_value += q_v
          q_value
        return q_value 

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
          return None 
        else: 
          best_action = None
          best_value = -float('inf')
          for action in self.mdp.getPossibleActions(state):
            value = self.computeQValueFromValues(state, action)
            if value > best_value: 
              best_action = action 
              best_value = value
          return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

      new_values = util.Counter()
      tracker = 0
      states = self.mdp.getStates()
      for i in range(self.iterations):
        if tracker >= len(states):
          tracker = 0
        action = self.computeActionFromValues(states[tracker])
        if action != None: 
          curr_val = self.computeQValueFromValues(states[tracker], action)
        else: 
          curr_val = 0
        new_values[states[tracker]] = curr_val
        tracker += 1
        self.values = new_values
       



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def predecessor(self, currstate):
      pred_set = set()
      allstates = self.mdp.getStates()
      for state in allstates:
          for action in self.mdp.getPossibleActions(state):
            if action != None:
              for succstate, prob in self.mdp.getTransitionStatesAndProbs(state, action): 
                if prob != 0 and succstate == currstate:
                  pred_set.add(state) 
      return pred_set

    def runValueIteration(self):
      statediff = util.PriorityQueue()
      for state in self.mdp.getStates():
        currsval = self.values[state]
        saction = self.computeActionFromValues(state)
        if saction != None:
          currsqval = self.computeQValueFromValues(state, saction)
        else: 
          currsqval = 0 
        diff = abs(currsval-currsqval)
        statediff.update(state, -diff)

      for i in range(self.iterations):
        if statediff.isEmpty():
          break
        else:
          currstate = statediff.pop()
          curraction = self.computeActionFromValues(currstate)
          if curraction != None: 
            currq = self.computeQValueFromValues(currstate, curraction)
          else:
            currq = 0 
          self.values[currstate] = currq
          for pred in self.predecessor(currstate):
            predval = self.values[pred]
            predaction = self.computeActionFromValues(pred)
            if predaction != None:
              predq = self.computeQValueFromValues(pred, predaction)
            else: 
              predq = 0 
            diff = abs(predval-predq)
            if diff > self.theta: 
              statediff.update(pred, -diff)