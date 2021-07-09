from copy import deepcopy
MAZE_H = 10  # height of the entire grid in cells
MAZE_W = 10  # width of the entire grid in cells

class Maze(object):
    def __init__(self, agentXY, goalXY, walls=[],pits=[]):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.wallblocks = set()
        self.pitblocks= set()
        self.MAZE_H = 10  # height of the entire grid in cells
        self.MAZE_W = 10  # width of the entire grid in cells
        self.build_shape_maze(agentXY, goalXY, walls, pits)

    def build_shape_maze(self,agentXY,goalXY, walls,pits):
        self.maze = [[0 for _ in range(MAZE_W)] for __ in range(MAZE_H)]
        # Walls
        for x,y in walls:
            self.wallblocks.add((x,y))
            self.maze[x][y] = -1
        # Pits
        for x,y in pits:
            self.pitblocks.add((x,y))
            self.maze[x][y] = -10

        # Goal
        self.goal = [goalXY[0],goalXY[1]]
        self.maze[goalXY[0]][goalXY[1]] = 1

        # Start
        self.agent = [agentXY[0],agentXY[1]]
        self.start = [agentXY[0],agentXY[1]]
    

    def maze_val(self, xy):
        return self.maze[xy[0]][xy[1]]


    def reset(self, value=1):
        self.agent[0] = self.start[0]
        self.agent[1] = self.start[1]
        return self.start


    def step(self, action):
        s = deepcopy(self.agent)

        if action == 0:   # up
            if self.agent[1] > 0:
                self.agent[1] -= 1
        elif action == 1:   # down
            if self.agent[1] < (MAZE_H - 1) :
                self.agent[1] += 1
        elif action == 2:   # right
            if self.agent[0] < (MAZE_W - 1) :
                self.agent[0] += 1
        elif action == 3:   # left
            if self.agent[0] > 0:
                self.agent[0] -= 1

        s_ = deepcopy(self.agent)
        reward, done = self.computeRewards(s, s_)

        return s_, reward, done


    def computeRewards(self, currstate, nextstate):
        mval = self.maze_val(nextstate) 
        # Goal
        if nextstate == self.goal:
            reward = 1
            done = True
            nextstate = 'terminal'
        # Wall
        elif mval == -1:
            reward = -0.3
            done = False
            nextstate = currstate
        # Pit
        elif mval == -10:
            reward = -10
            done = True
            nextstate = 'terminal'
        else:
            reward = -0.1
            done = False
        return reward,done


    def render(val=1):
        # Dummy function
        pass


    def destroy(val=1):
        # Dummy function
        pass

