import torch


class GridWorld:
    class CellType:
        FREE = 255
        WALL = 0
        TRAP = 50
        STRT = 100
        GOAL = 200

    class Action:
        UP=0
        LEFT=1
        DOWN=2
        RIGHT=3

    def __init__(self, width=10, height=10, drift_prob=0.1,
            move_cost=1, bump_cost=0, trap_cost=10, goal_reward=10):
        self.width = width
        self.height = height
        self.drift_prob = drift_prob
        self.move_cost = move_cost
        self.bump_cost = bump_cost
        self.trap_cost = trap_cost
        self.goal_reward = goal_reward
        self.cells = torch.full((height+2, width+2), GridWorld.CellType.FREE, dtype=torch.uint8)
        self.add_room(0,0,height+1,width+1)

    def add_horizontal_wall(self, at_y, from_x, to_x):
        self.cells[at_y,from_x:to_x+1] = GridWorld.CellType.WALL

    def add_vertical_wall(self, at_x, from_y, to_y):
        self.cells[from_y:to_y+1, at_x] = GridWorld.CellType.WALL

    def add_room(self, top, left, bot, right):
        self.add_horizontal_wall(top, left, right)
        self.add_horizontal_wall(bot, left, right)
        self.add_vertical_wall(left, top+1, bot-1)
        self.add_vertical_wall(right, top+1, bot-1)

    def add_start(self, at_x, at_y):
        self.cells[at_y, at_x] = GridWorld.CellType.STRT

    def add_trap(self, at_x, at_y):
        self.cells[at_y, at_x] = GridWorld.CellType.TRAP

    def add_goal(self, at_x, at_y):
        self.cells[at_y, at_x] = GridWorld.CellType.GOAL

    def clear_cell(self, at_x, at_y):
        self.cells[at_y, at_x] = GridWorld.CellType.FREE

    def __str__(self):
        if hasattr(self, "agent_x"):
            content = self.cells.clone()
            content[self.agent_y, self.agent_x] = 250
        else:
            content = self.cells
        return '\n'.join(' '.join("{}" for x in range(self.width+2)) for y in range(self.height+2)).format(*({
            GridWorld.CellType.WALL: "X",
            GridWorld.CellType.TRAP: "_",
            GridWorld.CellType.STRT: "o",
            GridWorld.CellType.GOAL: "$",
            250: "+"
            }.get(c.item(), " ") for c in content.view(-1)))

    @property
    def state_space_size(self):
        return self.width * self.height

    @property
    def action_space_size(self):
        return 4

    @property
    def transition_tensor(self):
        if not hasattr(self, "_t"):
            self._t = torch.zeros((self.state_space_size, self.action_space_size, self.state_space_size))
            for x in range(self.width):
                for y in range(self.height):
                    s = self.width * y + x
                    if self.cells[y,x+1] == GridWorld.CellType.WALL:
                        self._t[s, GridWorld.Action.UP, s] = 1 - self.drift_prob
                        self._t[s, GridWorld.Action.LEFT, s] = 0.5*self.drift_prob
                        self._t[s, GridWorld.Action.RIGHT, s] = 0.5*self.drift_prob
                    else:
                        self._t[s, GridWorld.Action.UP, self.width * (y-1) + x] = 1 - self.drift_prob
                        self._t[s, GridWorld.Action.LEFT, self.width * (y-1) + x] = 0.5*self.drift_prob
                        self._t[s, GridWorld.Action.RIGHT, self.width * (y-1) + x] = 0.5*self.drift_prob

                    if self.cells[y+2,x+1] == GridWorld.CellType.WALL:
                        self._t[s, GridWorld.Action.DOWN, s] += 1 - self.drift_prob
                        self._t[s, GridWorld.Action.LEFT, s] += 0.5*self.drift_prob
                        self._t[s, GridWorld.Action.RIGHT, s] += 0.5*self.drift_prob
                    else:
                        self._t[s, GridWorld.Action.DOWN, self.width * (y+1) + x] = 1 - self.drift_prob
                        self._t[s, GridWorld.Action.LEFT, self.width * (y+1) + x] = 0.5*self.drift_prob
                        self._t[s, GridWorld.Action.RIGHT, self.width * (y+1) + x] = 0.5*self.drift_prob

                    if self.cells[y+1,x] == GridWorld.CellType.WALL:
                        self._t[s, GridWorld.Action.LEFT, s] += 1 - self.drift_prob
                        self._t[s, GridWorld.Action.UP, s] += 0.5*self.drift_prob
                        self._t[s, GridWorld.Action.DOWN, s] += 0.5*self.drift_prob
                    else:
                        self._t[s, GridWorld.Action.LEFT, self.width * y + x - 1] = 1 - self.drift_prob
                        self._t[s, GridWorld.Action.UP, self.width * y + x - 1] = 0.5*self.drift_prob
                        self._t[s, GridWorld.Action.DOWN, self.width * y + x - 1] = 0.5*self.drift_prob

                    if self.cells[y+1,x+2] == GridWorld.CellType.WALL:
                        self._t[s, GridWorld.Action.RIGHT, s] += 1 - self.drift_prob
                        self._t[s, GridWorld.Action.UP, s] += 0.5*self.drift_prob
                        self._t[s, GridWorld.Action.DOWN, s] += 0.5*self.drift_prob
                    else:
                        self._t[s, GridWorld.Action.RIGHT, self.width * y + x + 1] = 1 - self.drift_prob
                        self._t[s, GridWorld.Action.UP, self.width * y + x + 1] = 0.5*self.drift_prob
                        self._t[s, GridWorld.Action.DOWN, self.width * y + x + 1] = 0.5*self.drift_prob
        return self._t

    @property
    def reward_tensor(self):
        if not hasattr(self, "_r"):
            self._r = torch.full((self.state_space_size, self.action_space_size, self.state_space_size),
                    -self.move_cost)
            srange = torch.arange(self.state_space_size)
            self._r[srange, :, srange] -= self.bump_cost
            self._r[:, :, self.cells[1:-1,1:-1].reshape(-1) == GridWorld.CellType.TRAP] -= self.trap_cost
            self._r[:, :, self.cells[1:-1,1:-1].reshape(-1) == GridWorld.CellType.GOAL] += self.goal_reward
        return self._r

    @property
    def terminal_state_mask(self):
        if not hasattr(self, "_m"):
            self._m = (self.cells[1:-1, 1:-1] == GridWorld.CellType.TRAP) \
                    | (self.cells[1:-1, 1:-1] == GridWorld.CellType.GOAL)
        return self._m.view(-1)

    def reset(self):
        start_pos = (self.cells == GridWorld.CellType.STRT).nonzero()
        self.agent_y, self.agent_x = start_pos[torch.randint(0,start_pos.size(0), (1,))].squeeze(0).tolist()
        return self.width * (self.agent_y - 1) + self.agent_x - 1

    def step(self, action):
        drift = torch.rand((1,))
        if drift < 0.5*self.drift_prob:
            action = (action - 1) % self.action_space_size
        elif drift < self.drift_prob:
            action = (action + 1) % self.action_space_size

        bump = True
        if action == GridWorld.Action.UP \
                and self.cells[self.agent_y - 1, self.agent_x] != GridWorld.CellType.WALL:
                    self.agent_y -= 1
                    bump = False
        elif action == GridWorld.Action.LEFT \
                and self.cells[self.agent_y, self.agent_x - 1] != GridWorld.CellType.WALL:
                    self.agent_x -= 1
                    bump = False
        elif action == GridWorld.Action.DOWN \
                and self.cells[self.agent_y + 1, self.agent_x] != GridWorld.CellType.WALL:
                    self.agent_y += 1
                    bump = False
        elif action == GridWorld.Action.RIGHT \
                and self.cells[self.agent_y, self.agent_x + 1] != GridWorld.CellType.WALL:
                    self.agent_x += 1
                    bump = False

        reward = -self.move_cost
        if bump:
            reward -= self.bump_cost

        done = False
        if self.cells[self.agent_y, self.agent_x] == GridWorld.CellType.TRAP:
            reward -= self.trap_cost
            done = True
        elif self.cells[self.agent_y, self.agent_x] == GridWorld.CellType.GOAL:
            reward += self.goal_reward
            done = True
        return self.width * (self.agent_y - 1) + self.agent_x - 1, reward, done
