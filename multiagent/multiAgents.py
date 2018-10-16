# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # we always want to be at least 1 away from any ghosts unless we can eat them
        if min([2] + [util.manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates if
                      ghost.scaredTimer == 0]) < 2:
            return -1000000

        # punish pacman for not moving 
        if action == Directions.STOP:
            return -1000000

        # we want pacman to eat pellets and ghosts if he can
        pellets = currentGameState.getFood().asList()
        distanceToClosestPellet = min([util.manhattanDistance(pellet, newPos) for pellet in
                                       pellets + [ghost.getPosition() for ghost in newGhostStates if
                                                  ghost.scaredTimer != 0]]) if len(
            pellets) != 1 else util.manhattanDistance(pellets[0], newPos)
        return -distanceToClosestPellet  # (1.0 / distanceToClosestPellet / distanceToGhost)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def minimax(state, depth, agent):

            # check if depth is 0 or if we are at an end state
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            # we want to maximize pacman
            if agent == 0:
                value = float("-inf")
                best_action = None

                for action in state.getLegalActions(0):
                    temp_score, _ = minimax(state.generateSuccessor(0, action),
                                            depth,
                                            agent + 1 if agent + 1 < gameState.getNumAgents() else 0)
                    if temp_score > value:
                        value, best_action = temp_score, action
                return value, best_action

            # and minimize other agents, aka ghosts
            else:
                value = float("inf")
                best_action = None

                for action in state.getLegalActions(agent):
                    temp_score, _ = minimax(state.generateSuccessor(agent, action),
                                            depth - 1 if agent == gameState.getNumAgents() - 1 else depth,
                                            agent + 1 if agent + 1 < gameState.getNumAgents() else 0)
                    if temp_score < value:
                        value, best_action = temp_score, action
                return value, best_action

        return minimax(gameState, self.depth, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"

        def alphabeta(state, depth, agent, alpha, beta):

            # check if we are at an end state
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # we want to maximize pacman
            if agent == 0:
                value = float("-inf")
                best_action = Directions.STOP
                for action in state.getLegalActions(agent):
                    temp_score = alphabeta(state.generateSuccessor(agent, action),
                                           depth,
                                           agent + 1,
                                           alpha,
                                           beta)
                    if temp_score > value:
                        value, best_action = temp_score, action
                    alpha = max(alpha, value)
                    if value > beta:
                        return value
                if depth == 0:
                    return best_action
                return value

            # and minimize other agents, aka ghosts
            else:
                value = float("inf")
                next_agent = agent + 1 if agent + 1 < gameState.getNumAgents() else 0
                for action in state.getLegalActions(agent):
                    # if we are at the last agent check if we have reached max depth
                    if next_agent == 0 and depth == self.depth - 1:
                        temp_score = self.evaluationFunction(state.generateSuccessor(agent, action))
                    else:
                        temp_score = alphabeta(state.generateSuccessor(agent, action),
                                               depth + 1 if next_agent == 0 else depth,
                                               next_agent,
                                               alpha,
                                               beta)
                    if temp_score < value:
                        value = temp_score
                    beta = min(beta, value)
                    if value < alpha:
                        return value
                return value

        return alphabeta(gameState, 0, 0, float("-inf"), float("inf"))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(state, depth, agent):

            # check if we are at an end state
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # we want to maximize pacman
            if agent == 0:
                value = float("-inf")
                best_action = Directions.STOP
                next_agent = agent + 1

                for action in state.getLegalActions(agent):

                    temp_score = expectimax(state.generateSuccessor(agent, action),
                                            depth,
                                            next_agent)

                    if temp_score > value:
                        value = temp_score
                        best_action = action

                if depth == 0:
                    return best_action

                return value

            # and minimize other agents, aka ghosts
            else:
                value = 0
                next_agent = agent + 1 if agent + 1 < gameState.getNumAgents() else 0

                for action in state.getLegalActions(agent):

                    prob = 1.0 / len(state.getLegalActions(agent))

                    # if we are at the last agent check if we have reached max depth
                    if next_agent == 0 and depth == self.depth - 1:
                        value += self.evaluationFunction(state.generateSuccessor(agent, action)) * prob

                    else:
                        value += expectimax(state.generateSuccessor(agent, action),
                                           depth + 1 if next_agent == 0 else depth,
                                           next_agent) * prob

                return value

        return expectimax(gameState, 0, 0)




def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:

      If we won, return high score, if we lost return wost score.

      Calculate distance to closest ghost and reward pacman for staying at least 1 step away from them.
      Calculate distance to closest pellet or edible ghost and penalize pacman for being far away from it.
      Penalize pacman for the amount of food left to encourage him to eat more.
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    pellets = currentGameState.getFood().asList()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")
    
    # we always want to be at least 1 away from any ghosts unless we can eat them
    distanceToClosestGhost = max([1] +[util.manhattanDistance(ghost.getPosition(), newPos) for ghost in newGhostStates if ghost.scaredTimer == 0])

    # we want pacman to eat pellets and ghosts if he can
    distanceToClosestPellet = min([util.manhattanDistance(pellet, newPos) for pellet in pellets +
                                  [ghost.getPosition() for ghost in newGhostStates if ghost.scaredTimer != 0]]) \
                                  if len(pellets) != 1 else util.manhattanDistance(pellets[0], newPos)
    
    # if pacman can eat something this is great
    if distanceToClosestPellet == 0:
        return float("inf")

    foodLeft = currentGameState.getNumFood()

    pelletWeight, ghostWeight, foodWeight = 2, 2, 2

    # if the ghost is far away then we don't care
    if distanceToClosestGhost > 2:
        ghostWeight = 0

    return (scoreEvaluationFunction(currentGameState)) + (-distanceToClosestPellet * pelletWeight) + (distanceToClosestGhost * ghostWeight) + (-foodLeft * foodWeight)

# Abbreviation
better = betterEvaluationFunction