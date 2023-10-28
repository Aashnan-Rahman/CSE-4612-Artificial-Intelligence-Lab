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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        foodDistances = [1e9]
        ghostDistances = [1e9]

        score = 0
        TotalFoods = len(currentGameState.getFood().asList())
        RemainingFood = TotalFoods - len(newFood.asList())
        
        for food in newFood.asList():
            foodDistances.append(util.manhattanDistance(newPos, food))
        
        closestFoodDistance = min(foodDistances)
    
        for ghost in newGhostStates:
            ghostDistances.append(util.manhattanDistance(newPos,ghost.getPosition()))
        
        closestGhostDistance = min(ghostDistances)
        
        
        score = 1.0/(closestFoodDistance + 1)
        
        weight1 = 1
        weight2 = 0
        weight3 = 0
        
        if closestGhostDistance <= 1:
            weight2 = -1e9

        elif RemainingFood > 0:
            weight3 = 1e9
        
        score = ((1.0/(closestFoodDistance + 1)) * weight1) + ((1.0/(closestGhostDistance+1))*weight2) + (RemainingFood*weight3)
        
        return score
        #return successorGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.value(gameState, 0, self.depth)[1]

    def value(self, gameState, agentIndex, depth):

        if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP
        
        elif agentIndex == 0:
            return self.maximizer(gameState, agentIndex, depth)
        
        else:
            return self.minimizer(gameState, agentIndex, depth)
        
    def minimizer(self, gameState, agentIndex, depth):
        
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        
        if nextAgent == 0:
            nextDepth = depth - 1
        
        else:
            nextDepth = depth

        currentScore, currentAction = 1e9, Directions.STOP
        actionList = gameState.getLegalActions(agentIndex)

        for action in actionList:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorScore = self.value(successorGameState, nextAgent, nextDepth)[0]
            
            if currentScore > successorScore:
                currentScore = successorScore
                currentAction = action

        return currentScore, currentAction
    
    def maximizer(self, gameState, agentIndex, depth):
        
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        
        if nextAgent == 0:
            nextDepth = depth - 1
        
        else:
            nextDepth = depth

        currentScore, currentAction = -1e9, Directions.STOP
        actionList = gameState.getLegalActions(agentIndex)

        for action in actionList:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorScore = self.value(successorGameState, nextAgent, nextDepth)[0]
            
            if currentScore < successorScore:
                currentScore = successorScore
                currentAction = action

        return currentScore, currentAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        return self.value(gameState, 0, self.depth,-1e9,1e9)[1]

    def value(self, gameState, agentIndex, depth,a,b):
        
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP
        
        elif agentIndex == 0:
            return self.maximizer(gameState, agentIndex, depth,a,b)
        
        else:
            return self.minimizer(gameState, agentIndex, depth,a,b)
        
    def minimizer(self, gameState, agentIndex, depth,a,b):
        
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        
        if nextAgent == 0:
            nextDepth = depth - 1
        
        else:
            nextDepth = depth

        currentScore, currentAction = 1e9, Directions.STOP
        actionList = gameState.getLegalActions(agentIndex)

        for action in actionList:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorScore = self.value(successorGameState, nextAgent, nextDepth,a,b)[0]
            
            if currentScore > successorScore:
                currentScore = successorScore
                currentAction = action
            
            if currentScore < a:
                return currentScore, currentAction
            
            b = min(currentScore,b)

        return currentScore, currentAction
    
    def maximizer(self, gameState, agentIndex, depth,a,b):
        
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        
        if nextAgent == 0:
            nextDepth = depth - 1
        
        else:
            nextDepth = depth

        currentScore, currentAction = -1e9, Directions.STOP
        actionList = gameState.getLegalActions(agentIndex)

        for action in actionList:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorScore = self.value(successorGameState, nextAgent, nextDepth,a,b)[0]
            
            if currentScore < successorScore:
                currentScore = successorScore
                currentAction = action
            
            if currentScore > b:
                return currentScore, currentAction

            a = max(currentScore,a)

        return currentScore, currentAction


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
        #util.raiseNotDefined()
        return self.value(gameState, 0, self.depth)[1]

    def value(self, gameState, agentIndex, depth):
        
        if gameState.isLose() or gameState.isWin() or depth == 0:
            return self.evaluationFunction(gameState), Directions.STOP
        
        elif agentIndex == 0:
            return self.maximizer(gameState, agentIndex, depth)
        
        else:
            return self.expectation(gameState, agentIndex, depth)
        
    def expectation(self, gameState, agentIndex, depth):
        
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        
        if nextAgent == 0:
            nextDepth = depth - 1
        
        else:
            nextDepth = depth

        
        actionList = gameState.getLegalActions(agentIndex)
        expected_value = 0
        probability = 1/len(actionList)

        for action in actionList:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorScore = self.value(successorGameState, nextAgent, nextDepth)[0]
            
            expected_value += (probability * successorScore)
        
        currentScore = expected_value
        currentAction = random.choice(actionList)

        return currentScore, currentAction
    
    def maximizer(self, gameState, agentIndex, depth):
        
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        
        if nextAgent == 0:
            nextDepth = depth - 1
        
        else:
            nextDepth = depth

        currentScore, currentAction = -1e9, Directions.STOP
        actionList = gameState.getLegalActions(agentIndex)

        for action in actionList:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            successorScore = self.value(successorGameState, nextAgent, nextDepth)[0]
            
            if currentScore < successorScore:
                currentScore = successorScore
                currentAction = action

        return currentScore, currentAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()

    foodList = currentGameState.getFood().asList()
    foodCount = len(foodList)
    capsuleCount = len(currentGameState.getCapsules())
    closestFood = 1

    gameScore = currentGameState.getScore()

    food_distances = [manhattanDistance(pacmanPosition, foodPosition) for foodPosition in foodList]

    if foodCount > 0:
        closestFood = min(food_distances)

    for ghost_position in ghostPositions:
        ghost_distance = manhattanDistance(pacmanPosition, ghost_position)

        if ghost_distance < 2:
            closestFood = 100000


    weight1 = 2000
    weight2 = 500
    weight3 = -250
    weight4 = -100


    value = (1.0 / closestFood)*weight1 + (gameScore*weight2) + (foodCount*weight3) + (capsuleCount*weight4)

    return value

# Abbreviation
better = betterEvaluationFunction
