"""
LNS-wPBS Multi-Agent Pickup and Delivery (MAPD) Solver
Based on "Multi-Goal Multi-Agent Pickup and Delivery" (Xu et al., 2022)
and "Lifelong Multi-Agent Path Finding in Large-Scale Warehouses" (Li et al., 2021)
"""

import csv
import heapq
import os
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time

# Constants
GRID_SIZE = 20
INFINITY = float('inf')
DEFAULT_WINDOW = 20  # Increased for better collision resolution
DEFAULT_REPLAN_PERIOD = 5  # Reduced for more responsive task assignment

class Orientation(Enum):
    EAST = 0    # +X
    NORTH = 90  # +Y  
    WEST = 180  # -X
    SOUTH = 270 # -Y

class Action(Enum):
    WAIT = 'wait'
    MOVE = 'move'
    TURN_LEFT = 'turn_left'  # 90 degrees CCW
    TURN_RIGHT = 'turn_right'  # 90 degrees CW
    TURN_AROUND = 'turn_around'  # 180 degrees
    PICKUP = 'pickup'
    DROPOFF = 'dropoff'

@dataclass
class Conflict:
    """Represents a conflict between two agents"""
    agent1: str
    agent2: str
    location: Any  # Can be tuple for vertex or tuple of tuples for edge
    timestep: int
    type: str  # 'vertex' or 'edge'

@dataclass
class Constraint:
    """Represents a constraint for an agent"""
    agent: str
    location: Any  # Can be tuple for vertex or tuple of tuples for edge
    timestep: int
    type: str  # 'vertex' or 'edge'
    
@dataclass(frozen=False)
class Task:
    task_id: str
    start_point: str
    end_point: str
    priority: str
    remaining_time: Optional[int]
    pickup_location: Tuple[int, int] = None
    dropoff_locations: List[Tuple[int, int]] = field(default_factory=list)
    release_time: int = 0
    is_executing: bool = False  # Being executed by an AGV
    is_completed: bool = False  # Task has been completed
    assigned_to: Optional[str] = None  # Which AGV is assigned this task
    
    def __hash__(self):
        return hash(self.task_id)
    
    def __eq__(self, other):
        if isinstance(other, Task):
            return self.task_id == other.task_id
        return False

@dataclass
class AGV:
    name: str
    x: int
    y: int
    pitch: int
    home_x: int  # Initial/parking location
    home_y: int
    loaded: bool = False
    destination: str = ""
    current_task: Optional[Task] = None
    task_sequence: List[Task] = field(default_factory=list)
    path: List = field(default_factory=list)  # List of State objects
    path_index: int = 0
    returning_home: bool = False  # Track if AGV is returning to home
    waiting_at_cell: Optional[Tuple[int, int]] = None  # Track if AGV is at/heading to waiting cell
    last_action: Optional[Action] = None  # Track last action performed
    last_action_task: Optional[str] = None  # Task ID for last pickup/dropoff
    assigned_pickup_point: Optional[str] = None  # Which pickup point this AGV is assigned to

class State:
    """State for PBS path search with orientation"""
    def __init__(self, x: int, y: int, orientation: int, t: int, g: float = 0, h: float = 0):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.t = t
        self.g = g  # Cost so far
        self.h = h  # Heuristic
        self.f = g + h  # Total cost
        self.parent = None
        self.action = None
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return (self.x, self.y, self.orientation, self.t) == (other.x, other.y, other.orientation, other.t)
    
    def __hash__(self):
        return hash((self.x, self.y, self.orientation, self.t))

class PBSNode:
    """Node in PBS priority tree"""
    def __init__(self):
        self.priorities: List[Tuple[str, str]] = []  # (higher_priority_agent, lower_priority_agent)
        self.solution: Dict[str, List] = {}
        self.cost: int = 0
        
    def __lt__(self, other):
        return self.cost < other.cost

class LNSwPBSSolver:
    def __init__(self, window: int = DEFAULT_WINDOW, replan_period: int = DEFAULT_REPLAN_PERIOD):
        self.agvs: Dict[str, AGV] = {}
        self.tasks: List[Task] = []
        self.start_points: Dict[str, Tuple[int, int]] = {}
        self.end_points: Dict[str, Tuple[int, int]] = {}
        self.task_queues: Dict[str, deque] = defaultdict(deque)  # FIFO queues per pickup point
        self.obstacles: Set[Tuple[int, int]] = set()
        self.distance_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
        
        # Track which pickup points are occupied/assigned
        self.pickup_point_assignments: Dict[str, str] = {}  # pickup_point -> agent_name
        
        # Waiting cells for pre-positioning
        self.waiting_cells: Dict[str, List[Tuple[int, int]]] = {}  # pickup_point -> [waiting_positions]
        self.waiting_cell_reservations: Dict[Tuple[int, int], str] = {}  # waiting_cell -> agent_name
        
        # LNS-wPBS specific parameters
        self.window = window  # Time window for collision resolution
        self.replan_period = replan_period  # Timesteps between replanning
        self.current_timestep = 0
        self.last_replan_timestep = 0
        
    def load_data(self, map_file: str, task_file: str):
        """Load map and task data from CSV files"""
        # Load map data
        with open(map_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                type_val = row['type'].strip()
                name = row['name'].strip()
                x, y = int(row['x']), int(row['y'])
                
                if type_val == 'start_point':
                    self.start_points[name] = (x, y)
                    self.obstacles.add((x, y))
                elif type_val == 'end_point':
                    self.end_points[name] = (x, y)
                    self.obstacles.add((x, y))
                elif type_val == 'agv':
                    pitch = int(row['pitch'])
                    self.agvs[name] = AGV(name, x, y, pitch, x, y)
        
        # Calculate waiting cells for each pickup point
        for pickup_name, pickup_pos in self.start_points.items():
            pickup_loc = self.get_pickup_location(pickup_name)
            if pickup_loc:
                waiting_positions = []
                px, py = self.start_points[pickup_name]
                # Add waiting cells at +1Y and -1Y from pickup cell
                for dy in [1, -1]:
                    wx, wy = px, py + dy
                    if (1 <= wx <= GRID_SIZE and 1 <= wy <= GRID_SIZE and 
                        (wx, wy) not in self.obstacles):
                        waiting_positions.append((wx, wy))
                self.waiting_cells[pickup_name] = waiting_positions
        
        # Load tasks and organize by start point (FIFO queues)
        with open(task_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                task = Task(
                    task_id=row.get('task-id', row.get('task_id', '')),
                    start_point=row['start_point'].strip(),
                    end_point=row['end_point'].strip(),
                    priority=row['priority'],
                    remaining_time=None if row['remaining_time'] in ['None', ''] else int(row['remaining_time'])
                )
                
                # Calculate pickup and dropoff locations
                task.pickup_location = self.get_pickup_location(task.start_point)
                task.dropoff_locations = self.get_dropoff_locations(task.end_point)
                
                self.tasks.append(task)
                
                # Add to FIFO queue for this pickup point
                if task.pickup_location:
                    self.task_queues[task.start_point].append(task)
    
    def get_pickup_location(self, start_point: str) -> Optional[Tuple[int, int]]:
        """Get pickup location based on start point"""
        if start_point not in self.start_points:
            print(f"Warning: Start point '{start_point}' not found in map data")
            return None
        x, y = self.start_points[start_point]
        
        # Special handling for left-side points
        if start_point in ["Tiger", "Dragon", "Horse"]:
            return (x + 1, y)
        else:
            # Right-side points
            return (x - 1, y)
    
    def get_dropoff_locations(self, end_point: str) -> List[Tuple[int, int]]:
        """Get 4 possible dropoff locations for an endpoint"""
        if end_point not in self.end_points:
            return []
        x, y = self.end_points[end_point]
        locations = []
        
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE:
                if (nx, ny) not in self.obstacles:
                    locations.append((nx, ny))
        
        return locations
    
    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def compute_shortest_path_length(self, start: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Compute shortest path length between two points using BFS"""
        cache_key = (start, goal)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        if start == goal:
            return 0
        
        queue = deque([(start, 0)])
        visited = {start}
        
        while queue:
            (x, y), dist = queue.popleft()
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (nx, ny) == goal:
                    self.distance_cache[cache_key] = dist + 1
                    return dist + 1
                
                if (1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE and 
                    (nx, ny) not in self.obstacles and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append(((nx, ny), dist + 1))
        
        self.distance_cache[cache_key] = INFINITY
        return INFINITY
    
    def get_next_position(self, x: int, y: int, orientation: int) -> Tuple[int, int]:
        """Get next position if moving forward from current position and orientation"""
        if orientation == 0:  # East
            return (x + 1, y)
        elif orientation == 90:  # North
            return (x, y + 1)
        elif orientation == 180:  # West
            return (x - 1, y)
        elif orientation == 270:  # South
            return (x, y - 1)
        return (x, y)
    
    def get_best_waiting_cell(self, agent: AGV) -> Optional[Tuple[int, int]]:
        """
        Find the best available waiting cell for an agent based on task availability.
        """
        best_cell = None
        best_score = INFINITY
        
        for pickup_point, queue in self.task_queues.items():
            # Check if this pickup point has waiting tasks
            waiting_tasks = [t for t in queue if not t.is_executing and not t.is_completed]
            if not waiting_tasks:
                continue
            
            # Allow waiting near reserved pickup points (removed the check)
                
            # Check available waiting cells for this pickup point
            if pickup_point not in self.waiting_cells:
                continue
                
            for waiting_cell in self.waiting_cells[pickup_point]:
                # Check if cell is already reserved or occupied
                if waiting_cell in self.waiting_cell_reservations:
                    continue
                
                # Check if another agent is at this position
                occupied = False
                for other_agent in self.agvs.values():
                    if other_agent != agent and (other_agent.x, other_agent.y) == waiting_cell:
                        occupied = True
                        break
                
                if occupied:
                    continue
                
                # Calculate distance to this waiting cell
                dist = self.compute_shortest_path_length((agent.x, agent.y), waiting_cell)
                if dist == INFINITY:
                    continue
                
                # Score based on distance and queue length
                queue_bonus = len(waiting_tasks) * 5  # Prefer pickups with more tasks
                score = dist - queue_bonus
                
                if score < best_score:
                    best_score = score
                    best_cell = waiting_cell
        
        return best_cell
    
    def clear_stale_waiting_reservations(self):
        """
        Clear waiting cell reservations for agents that have moved on
        """
        for cell, agent_name in list(self.waiting_cell_reservations.items()):
            agent = self.agvs[agent_name]
            # Clear if agent has a task or is no longer heading to this cell
            if agent.current_task or agent.task_sequence or agent.waiting_at_cell != cell:
                del self.waiting_cell_reservations[cell]
                if agent.waiting_at_cell == cell:
                    agent.waiting_at_cell = None
    
    def get_next_available_task(self, pickup_point: str) -> Optional[Task]:
        """Get the next available task from a pickup point's FIFO queue"""
        if pickup_point not in self.task_queues:
            return None
        
        queue = self.task_queues[pickup_point]
        # Find first task that is not executing and not completed
        for task in queue:
            if not task.is_executing and not task.is_completed:
                return task
        return None

    def _eligible_front_tasks(self) -> List[Tuple[str, Task]]:
        """
        Return [(pickup_point, front_task)] for all pickups that are NOT currently locked
        by pickup_point_assignments and whose front task is available.
        """
        elig = []
        for pickup_point, queue in self.task_queues.items():
            # skip locked pickups (you release on PICKUP)
            if pickup_point in self.pickup_point_assignments:
                continue
            t = self.get_next_available_task(pickup_point)
            if t and t.pickup_location:
                elig.append((pickup_point, t))
        return elig

    def _build_cost_matrix(self, free_agents: List[AGV], tasks: List[Task]) -> List[List[int]]:
        """
        Build cost matrix C[i][j] = shortest path length (agent_i -> task_j.pickup_location).
        Unreachable = big M.
        """
        BIG_M = GRID_SIZE * GRID_SIZE * 100  # safe large number
        C = []
        for a in free_agents:
            row = []
            for t in tasks:
                d = self.compute_shortest_path_length((a.x, a.y), t.pickup_location)
                row.append(BIG_M if d == INFINITY else d)
            C.append(row)
        return C

    def _hungarian(self, cost: List[List[int]]) -> List[Tuple[int, int]]:
        """
        Hungarian algorithm (min-cost), O(n^3), supports rectangular matrices by padding.
        Returns list of (row_idx, col_idx) assignments within the original matrix bounds.
        """
        if not cost:
            return []
        n = len(cost)
        m = len(cost[0])
        n_ = max(n, m)

        # pad to square with BIG_M
        BIG_M = max(1_000_000, max(max(r) for r in cost) * 100 if cost and cost[0] else 1_000_000)
        a = [row + [BIG_M] * (n_ - m) for row in cost]
        for _ in range(n_ - n):
            a.append([BIG_M] * n_)

        # 1-based implementation
        u = [0] * (n_ + 1)
        v = [0] * (n_ + 1)
        p = [0] * (n_ + 1)
        way = [0] * (n_ + 1)

        for i in range(1, n_ + 1):
            p[0] = i
            j0 = 0
            minv = [float('inf')] * (n_ + 1)
            used = [False] * (n_ + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                delta = float('inf')
                j1 = 0
                for j in range(1, n_ + 1):
                    if not used[j]:
                        cur = a[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(0, n_ + 1):
                    if used[j]:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        # extract assignment
        assign_col_for_row = [-1] * n_
        for j in range(1, n_ + 1):
            if p[j] != 0:
                assign_col_for_row[p[j] - 1] = j - 1

        # only keep matches inside original bounds
        matches = []
        for i in range(n):
            j = assign_col_for_row[i]
            if j != -1 and j < m and cost[i][j] < BIG_M:
                matches.append((i, j))
        return matches

    def dynamic_task_assignment_lns(self):
        """
        Assign free agents to *front* tasks (FIFO) using Hungarian assignment.
        A pickup point is eligible only if it is not currently locked by pickup_point_assignments.
        Each eligible pickup contributes at most its front task to the assignment set.
        """
        print(f"Timestep {self.current_timestep}: Running dynamic task assignment...")

        # Clear stale pickup assignments and waiting reservations
        for pickup_point in list(self.pickup_point_assignments.keys()):
            agent_name = self.pickup_point_assignments[pickup_point]
            ag = self.agvs[agent_name]
            if not ag.current_task and not ag.task_sequence:
                del self.pickup_point_assignments[pickup_point]
                ag.assigned_pickup_point = None
        
        self.clear_stale_waiting_reservations()

        # Free agents = truly idle + not already locked to a pickup + agents returning home + agents at waiting cells
        free_agents = []
        for ag in self.agvs.values():
            if not ag.current_task and not ag.task_sequence:
                # Include agents returning home or at waiting cells as free agents
                if ag.assigned_pickup_point not in self.pickup_point_assignments:
                    free_agents.append(ag)
                    # If agent was returning home or waiting, cancel that status
                    if ag.returning_home:
                        ag.returning_home = False
                        print(f"  Agent {ag.name} interrupted return home for new assignment")
                    if ag.waiting_at_cell:
                        if ag.waiting_at_cell in self.waiting_cell_reservations:
                            if self.waiting_cell_reservations[ag.waiting_at_cell] == ag.name:
                                del self.waiting_cell_reservations[ag.waiting_at_cell]
                        ag.waiting_at_cell = None
                        print(f"  Agent {ag.name} leaving waiting cell for new assignment")

        candidates = self._eligible_front_tasks()  # [(pickup_point, task)]
        
        if free_agents and candidates:
            pickups = [p for p, _ in candidates]
            tasks = [t for _, t in candidates]

            # Build cost matrix and run Hungarian
            C = self._build_cost_matrix(free_agents, tasks)
            matches = self._hungarian(C)

            assignments_made = 0
            assigned_agents = set()
            for i, j in matches:
                agent = free_agents[i]
                task = tasks[j]
                pickup_point = pickups[j]

                # Double-check still eligible (race with other logic)
                if pickup_point in self.pickup_point_assignments:
                    continue
                if task.is_executing or task.is_completed or task.assigned_to:
                    continue

                # Lock this pickup to the agent until PICKUP happens
                self.pickup_point_assignments[pickup_point] = agent.name
                agent.assigned_pickup_point = pickup_point

                # Clear waiting cell reservation if agent had one
                if agent.waiting_at_cell and agent.waiting_at_cell in self.waiting_cell_reservations:
                    if self.waiting_cell_reservations[agent.waiting_at_cell] == agent.name:
                        del self.waiting_cell_reservations[agent.waiting_at_cell]
                agent.waiting_at_cell = None

                # Assign exactly one task (the front FIFO task)
                task.assigned_to = agent.name
                agent.task_sequence = [task]
                assigned_agents.add(agent.name)
                assignments_made += 1
                print(f"  Assigned 1 task from {pickup_point} to agent {agent.name} (distance: {C[i][j]})")

            print(f"  Made {assignments_made} new assignments")
            
            # Update free_agents list to exclude assigned agents
            free_agents = [ag for ag in free_agents if ag.name not in assigned_agents]
        else:
            if not candidates:
                print("  No eligible front tasks")
            if not free_agents:
                print("  No free agents")
        
        # ALWAYS try to assign remaining free agents to waiting cells
        # This includes the beginning when all pickup points are reserved
        remaining_free = [ag for ag in free_agents 
                         if not ag.waiting_at_cell and not ag.returning_home]
        
        waiting_assignments = 0
        home_assignments = 0
        
        for agent in remaining_free:
            best_cell = self.get_best_waiting_cell(agent)
            if best_cell:
                agent.waiting_at_cell = best_cell
                self.waiting_cell_reservations[best_cell] = agent.name
                waiting_assignments += 1
                print(f"  Agent {agent.name} assigned to wait at {best_cell}")
            else:
                # No waiting cells available, return home
                agent.returning_home = True
                home_assignments += 1
                print(f"  Agent {agent.name} returning home (no waiting cells available)")
        
        if waiting_assignments > 0:
            print(f"  Assigned {waiting_assignments} agents to waiting cells")
        if home_assignments > 0:
            print(f"  Sent {home_assignments} agents home")

    
    def get_all_agent_positions(self, at_time: int = None) -> Dict[str, Tuple[int, int]]:
        """Get positions of all agents at a specific time or current positions"""
        positions = {}
        for name, agent in self.agvs.items():
            if at_time is None:
                positions[name] = (agent.x, agent.y)
            else:
                # Find position at specific time from path
                time_offset = at_time - self.current_timestep
                if 0 <= agent.path_index + time_offset < len(agent.path):
                    state = agent.path[agent.path_index + time_offset]
                    positions[name] = (state.x, state.y)
                else:
                    # Agent will be at end of path or current position
                    if agent.path and agent.path_index < len(agent.path):
                        state = agent.path[-1]
                        positions[name] = (state.x, state.y)
                    else:
                        positions[name] = (agent.x, agent.y)
        return positions
    
    def windowed_pbs_with_priorities(self, priorities: List[Tuple[str, str]] = None) -> Dict[str, List[State]]:
        """
        Windowed Priority-Based Search for path planning.
        Only resolves collisions within the time window.
        """
        paths = {}
        
        if priorities is None:
            priorities = []
        
        # Build priority graph
        higher_priority = defaultdict(set)
        lower_priority = defaultdict(set)
        for high, low in priorities:
            higher_priority[low].add(high)
            lower_priority[high].add(low)
        
        # Topological sort to determine planning order
        def get_planning_order():
            order = []
            visited = set()
            
            def visit(agent):
                if agent in visited:
                    return
                visited.add(agent)
                # Visit all agents that must be planned before this one
                for other in higher_priority.get(agent, []):
                    visit(other)
                order.append(agent)
            
            for agent_name in self.agvs:
                visit(agent_name)
            
            return order
        
        planning_order = get_planning_order()
        
        # Plan paths in the determined order
        for agent_name in planning_order:
            agent = self.agvs[agent_name]
            
            # Get constraints from all higher priority agents
            constraints = []
            
            for other_name in higher_priority.get(agent_name, []):
                if other_name in paths:
                    other_path = paths[other_name]
                    for i, state in enumerate(other_path):
                        # Add constraints for entire path within window
                        if state.t - self.current_timestep < self.window:
                            # Vertex constraint
                            constraints.append(Constraint(
                                agent_name,
                                (state.x, state.y),
                                state.t,
                                'vertex'
                            ))
                            
                            # Edge constraint
                            if i > 0:
                                prev_state = other_path[i-1]
                                if prev_state.t == state.t - 1:
                                    # Prevent swapping
                                    constraints.append(Constraint(
                                        agent_name,
                                        ((state.x, state.y), (prev_state.x, prev_state.y)),
                                        prev_state.t,
                                        'edge'
                                    ))
                                    # Also prevent traversing same edge
                                    constraints.append(Constraint(
                                        agent_name,
                                        ((prev_state.x, prev_state.y), (state.x, state.y)),
                                        prev_state.t,
                                        'edge'
                                    ))
            
            # Get current positions of all agents as obstacles
            static_obstacles = set()
            
            # Add home positions of all other agents
            for other_name, other_agent in self.agvs.items():
                if other_name != agent_name:
                    static_obstacles.add((other_agent.home_x, other_agent.home_y))
            
            # Add current positions of idle agents and agents at waiting cells
            for other_name, other_agent in self.agvs.items():
                if other_name != agent_name:
                    if not other_agent.current_task and not other_agent.task_sequence:
                        # Add current position
                        static_obstacles.add((other_agent.x, other_agent.y))
                        
                        # If agent is at a waiting cell, add it as static obstacle
                        if other_agent.waiting_at_cell:
                            static_obstacles.add(other_agent.waiting_at_cell)
            
            # Plan path for this agent
            path = self.plan_windowed_path(agent, constraints, static_obstacles)
            if path:
                paths[agent_name] = path
            else:
                # If no path found, keep agent at current position
                paths[agent_name] = [State(agent.x, agent.y, agent.pitch, self.current_timestep)]
        
        return paths
    
    def plan_windowed_path(self, agent: AGV, constraints: List[Constraint], 
                           static_obstacles: Set[Tuple[int, int]]) -> List[State]:
        """
        Plan path for agent through its task sequence with windowed collision checking.
        Returns agent to waiting cell or home after completing all tasks.
        """
        full_path = []
        current_pos = (agent.x, agent.y, agent.pitch)
        current_time = self.current_timestep
        
        # If agent has a current task, continue executing it
        if agent.current_task:
            task = agent.current_task
            
            # If not loaded, go to pickup
            if not agent.loaded and task.pickup_location:
                pickup_path = self.a_star_windowed(
                    current_pos,
                    task.pickup_location,
                    current_time,
                    constraints,
                    static_obstacles
                )
                if pickup_path:
                    full_path.extend(pickup_path[1:])
                    if full_path:
                        last_state = full_path[-1]
                        current_pos = (last_state.x, last_state.y, last_state.orientation)
                        current_time = last_state.t
                    
                    # Add pickup action
                    pickup_state = State(current_pos[0], current_pos[1], current_pos[2], current_time + 1)
                    pickup_state.action = Action.PICKUP
                    full_path.append(pickup_state)
                    current_time += 1
            
            # If loaded or just picked up, go to dropoff
            if task.dropoff_locations:
                best_dropoff_path = None
                best_cost = INFINITY
                
                for dropoff_loc in task.dropoff_locations:
                    dropoff_path = self.a_star_windowed(
                        current_pos,
                        dropoff_loc,
                        current_time,
                        constraints,
                        static_obstacles
                    )
                    
                    if dropoff_path and len(dropoff_path) < best_cost:
                        best_cost = len(dropoff_path)
                        best_dropoff_path = dropoff_path
                
                if best_dropoff_path:
                    full_path.extend(best_dropoff_path[1:])
                    if full_path:
                        last_state = full_path[-1]
                        current_pos = (last_state.x, last_state.y, last_state.orientation)
                        current_time = last_state.t
                    
                    # Add dropoff action
                    dropoff_state = State(current_pos[0], current_pos[1], current_pos[2], current_time + 1)
                    dropoff_state.action = Action.DROPOFF
                    full_path.append(dropoff_state)
                    current_time += 1
        
        # If no current task but has tasks in sequence, go to first task's pickup
        elif agent.task_sequence:
            task = agent.task_sequence[0]
            if task.pickup_location:
                pickup_path = self.a_star_windowed(
                    current_pos,
                    task.pickup_location,
                    current_time,
                    constraints,
                    static_obstacles
                )
                if pickup_path:
                    full_path.extend(pickup_path[1:])
        
        # If agent has a waiting cell target, go there
        elif agent.waiting_at_cell:
            if (agent.x, agent.y) != agent.waiting_at_cell:
                waiting_path = self.a_star_windowed(
                    current_pos,
                    agent.waiting_at_cell,
                    current_time,
                    constraints,
                    static_obstacles
                )
                if waiting_path:
                    full_path.extend(waiting_path[1:])
                    if full_path:
                        last_state = full_path[-1]
                        current_pos = (last_state.x, last_state.y, last_state.orientation)
                        current_time = last_state.t
        
        # If returning home or no tasks and no waiting cell, go to home position
        elif agent.returning_home or (not agent.current_task and not agent.task_sequence and not agent.waiting_at_cell):
            if (agent.x, agent.y) != (agent.home_x, agent.home_y):
                home_path = self.a_star_windowed(
                    current_pos,
                    (agent.home_x, agent.home_y),
                    current_time,
                    constraints,
                    static_obstacles
                )
                if home_path:
                    full_path.extend(home_path[1:])
                    if full_path:
                        last_state = full_path[-1]
                        current_pos = (last_state.x, last_state.y, last_state.orientation)
                        current_time = last_state.t
        
        # If agent is at target position (home or waiting cell) and has no tasks, stay there
        if not agent.current_task and not agent.task_sequence:
            target_pos = agent.waiting_at_cell if agent.waiting_at_cell else (agent.home_x, agent.home_y)
            if (agent.x, agent.y) == target_pos or (full_path and (full_path[-1].x, full_path[-1].y) == target_pos):
                # Extend path with wait actions at target position
                if full_path:
                    current_time = full_path[-1].t
                    target_x, target_y = target_pos
                    target_orient = full_path[-1].orientation if full_path else agent.pitch
                else:
                    target_x, target_y = target_pos
                    target_orient = agent.pitch
                    
                for t in range(current_time + 1, current_time + self.window + 1):
                    wait_state = State(target_x, target_y, target_orient, t)
                    wait_state.action = Action.WAIT
                    full_path.append(wait_state)
        
        # Ensure path extends to at least current_time + window
        if full_path:
            last_t = full_path[-1].t
            last_pos = (full_path[-1].x, full_path[-1].y, full_path[-1].orientation)
            while last_t < current_time + self.window:
                last_t += 1
                wait_state = State(last_pos[0], last_pos[1], last_pos[2], last_t)
                wait_state.action = Action.WAIT
                full_path.append(wait_state)
        else:
            # No path generated, create wait actions
            for t in range(current_time + 1, current_time + self.window + 1):
                wait_state = State(current_pos[0], current_pos[1], current_pos[2], t)
                wait_state.action = Action.WAIT
                full_path.append(wait_state)
        
        return full_path
    
    def a_star_windowed(self, start: Tuple[int, int, int], 
                       goal: Tuple[int, int],
                       start_time: int,
                       constraints: List[Constraint],
                       static_obstacles: Set[Tuple[int, int]]) -> List[State]:
        """
        A* search with windowed collision checking.
        Continues planning beyond window but ignores constraints after window.
        """
        initial = State(start[0], start[1], start[2], start_time)
        initial.h = self.manhattan_distance((start[0], start[1]), goal)
        initial.f = initial.g + initial.h
        
        open_list = [initial]
        closed_set = set()
        
        # Build constraint tables (only for constraints within window)
        vertex_constraints = set()
        edge_constraints = set()
        
        for c in constraints:
            if c.type == 'vertex':
                vertex_constraints.add((c.location, c.timestep))
            elif c.type == 'edge':
                edge_constraints.add((c.location, c.timestep))
        
        # Limit search depth to prevent infinite loops
        max_time = start_time + 200
        
        while open_list and open_list[0].t < max_time:
            current = heapq.heappop(open_list)
            
            if (current.x, current.y) == goal:
                # Reconstruct path
                path = []
                while current:
                    path.append(current)
                    current = current.parent
                return list(reversed(path))
            
            state_key = (current.x, current.y, current.orientation, current.t)
            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            
            # Check if we're within the constraint window
            within_window = (current.t - self.current_timestep) < self.window
            
            # Generate successors
            successors = []
            
            # Wait
            next_pos = (current.x, current.y)
            can_wait = True
            if within_window and ((next_pos, current.t + 1) in vertex_constraints):
                can_wait = False
            
            if can_wait:
                wait_state = State(current.x, current.y, current.orientation, current.t + 1, current.g + 1)
                wait_state.h = self.manhattan_distance((current.x, current.y), goal)
                wait_state.f = wait_state.g + wait_state.h
                wait_state.parent = current
                wait_state.action = Action.WAIT
                successors.append(wait_state)
            
            # Move forward
            nx, ny = self.get_next_position(current.x, current.y, current.orientation)
            if (1 <= nx <= GRID_SIZE and 1 <= ny <= GRID_SIZE and 
                (nx, ny) not in self.obstacles and (nx, ny) not in static_obstacles):
                
                can_move = True
                if within_window:
                    # Check vertex constraint
                    if ((nx, ny), current.t + 1) in vertex_constraints:
                        can_move = False
                    # Check edge constraint (swapping) - MORE ROBUST checking
                    elif (((nx, ny), (current.x, current.y)), current.t) in edge_constraints:
                        can_move = False
                    # Also check reverse direction to be thorough
                    elif (((current.x, current.y), (nx, ny)), current.t) in edge_constraints:
                        can_move = False
                
                if can_move:
                    move_state = State(nx, ny, current.orientation, current.t + 1, current.g + 1)
                    move_state.h = self.manhattan_distance((nx, ny), goal)
                    move_state.f = move_state.g + move_state.h
                    move_state.parent = current
                    move_state.action = Action.MOVE
                    successors.append(move_state)
            
            # Turns (only check vertex constraints and prevent consecutive turns)
            # Check if parent action was a turn
            is_parent_turn = False
            if current.parent and current.action in [Action.TURN_LEFT, Action.TURN_RIGHT, Action.TURN_AROUND]:
                is_parent_turn = True
            
            if not is_parent_turn:  # Only add turn actions if parent wasn't a turn
                for new_orient, action in [
                    ((current.orientation + 90) % 360, Action.TURN_LEFT),
                    ((current.orientation - 90) % 360, Action.TURN_RIGHT),
                    ((current.orientation + 180) % 360, Action.TURN_AROUND)
                ]:
                    can_turn = True
                    if within_window and ((current.x, current.y), current.t + 1) in vertex_constraints:
                        can_turn = False
                    
                    if can_turn:
                        turn_state = State(current.x, current.y, new_orient, current.t + 1, current.g + 1)
                        turn_state.h = self.manhattan_distance((current.x, current.y), goal)
                        turn_state.f = turn_state.g + turn_state.h
                        turn_state.parent = current
                        turn_state.action = action
                        successors.append(turn_state)
            
            # Add successors to open list
            for succ in successors:
                heapq.heappush(open_list, succ)
        
        return []  # No path found
    
    def find_windowed_conflicts(self, paths: Dict[str, List[State]]) -> List[Conflict]:
        """
        Find conflicts between paths within the time window.
        Checks both vertex and edge (swapping) conflicts comprehensively.
        """
        conflicts = []
        agents = list(paths.keys())
        
        # Need at least 2 agents to have conflicts
        if len(agents) < 2:
            return conflicts
        
        # Find earliest conflict
        earliest_conflict = None
        earliest_time = INFINITY
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                path1, path2 = paths[agent1], paths[agent2]
                
                if not path1 or not path2:
                    continue
                
                # Build time-indexed position maps for efficient lookup
                pos_map1 = {}  # time -> (x, y)
                pos_map2 = {}  # time -> (x, y)
                
                for state in path1:
                    if state.t - self.current_timestep < self.window:
                        pos_map1[state.t] = (state.x, state.y)
                
                for state in path2:
                    if state.t - self.current_timestep < self.window:
                        pos_map2[state.t] = (state.x, state.y)
                
                # Check for vertex conflicts
                for t in pos_map1:
                    if t in pos_map2:
                        if pos_map1[t] == pos_map2[t]:
                            if t < earliest_time:
                                earliest_time = t
                                earliest_conflict = Conflict(
                                    agent1, agent2,
                                    pos_map1[t],
                                    t,
                                    'vertex'
                                )
                
                # Check for edge conflicts (swapping)
                times = sorted(set(pos_map1.keys()) & set(pos_map2.keys()))
                for idx in range(len(times) - 1):
                    t1, t2 = times[idx], times[idx + 1]
                    if t2 == t1 + 1:  # Consecutive timesteps
                        pos1_t1 = pos_map1.get(t1)
                        pos1_t2 = pos_map1.get(t2)
                        pos2_t1 = pos_map2.get(t1)
                        pos2_t2 = pos_map2.get(t2)
                        
                        if pos1_t1 and pos1_t2 and pos2_t1 and pos2_t2:
                            # Check if they swap positions
                            if pos1_t1 == pos2_t2 and pos2_t1 == pos1_t2:
                                if t1 < earliest_time:
                                    earliest_time = t1
                                    earliest_conflict = Conflict(
                                        agent1, agent2,
                                        (pos1_t1, pos1_t2),
                                        t1,
                                        'edge'
                                    )
        
        if earliest_conflict:
            conflicts.append(earliest_conflict)
        
        return conflicts
    
    def pbs_high_level(self) -> Dict[str, List[State]]:
        """
        PBS high-level search with windowed collision resolution.
        More thorough conflict resolution with better pruning.
        """
        # Start with empty priority ordering
        root = PBSNode()
        root.solution = self.windowed_pbs_with_priorities([])
        
        # Calculate cost
        root.cost = sum(len(path) for path in root.solution.values())
        
        # Check if initial solution is conflict-free
        initial_conflicts = self.find_windowed_conflicts(root.solution)
        if not initial_conflicts:
            print(f"  PBS found conflict-free solution immediately with cost {root.cost}")
            return root.solution
        
        # DFS through priority tree
        stack = [root]
        best_solution = root.solution
        best_cost = root.cost
        iterations = 0
        max_iterations = 200  # Increased for more thorough search
        
        # Track visited priority orderings to avoid redundant work
        visited_priorities = set()
        
        print(f"  Starting PBS search...")
        
        while stack and iterations < max_iterations:
            iterations += 1
            node = stack.pop()
            
            # Skip if we've seen this priority ordering before
            priority_tuple = tuple(sorted(node.priorities))
            if priority_tuple in visited_priorities:
                continue
            visited_priorities.add(priority_tuple)
            
            # Find conflicts within window
            conflicts = self.find_windowed_conflicts(node.solution)
            
            if not conflicts:
                # No conflicts within window, solution found
                if node.cost < best_cost:
                    best_solution = node.solution
                    best_cost = node.cost
                print(f"  PBS found conflict-free solution with cost {node.cost} after {iterations} iterations")
                return node.solution
            
            # Choose first conflict
            conflict = conflicts[0]
            
            # Try both priority orderings
            for agent_order in [(conflict.agent1, conflict.agent2), 
                              (conflict.agent2, conflict.agent1)]:
                child = PBSNode()
                child.priorities = list(node.priorities)  # Copy parent priorities
                
                # Check if this creates a cycle
                has_cycle = False
                if agent_order in child.priorities:
                    has_cycle = True
                elif (agent_order[1], agent_order[0]) in child.priorities:
                    continue  # Skip opposite ordering
                else:
                    # Check for transitive cycles
                    higher = {agent_order[0]}
                    lower = {agent_order[1]}
                    
                    for high, low in child.priorities:
                        if low in higher:
                            higher.add(high)
                        if high in lower:
                            lower.add(low)
                    
                    if higher & lower:  # Intersection means cycle
                        has_cycle = True
                
                if not has_cycle:
                    child.priorities.append(agent_order)
                    
                    # Replan with new priorities
                    child.solution = self.windowed_pbs_with_priorities(child.priorities)
                    child.cost = sum(len(path) for path in child.solution.values())
                    
                    # Add to stack if promising (with tighter pruning)
                    if child.cost < best_cost * 1.2:  # Tighter bound
                        stack.append(child)
        
        print(f"  PBS completed {iterations} iterations, returning best solution with cost {best_cost}")
        return best_solution
    
    def execute_timestep(self):
        """
        Execute one timestep of the simulation.
        """
        self.current_timestep += 1
        
        # Track events
        task_completed = []
        task_picked_up = []
        agents_reached_home = []
        agents_reached_waiting = []
        
        # Update each agent
        for agent in self.agvs.values():
            # Clear last action from previous timestep
            agent.last_action = None
            agent.last_action_task = None
            
            if agent.path_index < len(agent.path):
                current_state = agent.path[agent.path_index]
                
                # Check for actions BEFORE moving
                if current_state.action == Action.PICKUP:
                    if agent.current_task:
                        agent.last_action = Action.PICKUP
                        agent.last_action_task = agent.current_task.task_id
                        agent.loaded = True
                        task_picked_up.append(agent.name)
                        print(f"  Agent {agent.name} picked up task {agent.current_task.task_id}")

                        # free this pickup point right after a successful pickup
                        pp = agent.current_task.start_point
                        if self.pickup_point_assignments.get(pp) == agent.name:
                            del self.pickup_point_assignments[pp]
                        agent.assigned_pickup_point = None
                
                elif current_state.action == Action.DROPOFF:
                    if agent.current_task:
                        agent.last_action = Action.DROPOFF
                        agent.last_action_task = agent.current_task.task_id
                        print(f"  Agent {agent.name} dropped off task {agent.current_task.task_id}")
                        
                        # Mark task as completed
                        agent.current_task.is_completed = True
                        agent.current_task.is_executing = False
                        agent.current_task.assigned_to = None
                        
                        # Remove from task queue
                        for queue in self.task_queues.values():
                            if agent.current_task in queue:
                                queue.remove(agent.current_task)
                                break
                        
                        task_completed.append(agent.name)
                        agent.current_task = None
                        agent.loaded = False
                        
                        # Check if agent has more tasks
                        if not agent.task_sequence:
                            # Try to find a waiting cell first
                            best_cell = self.get_best_waiting_cell(agent)
                            if best_cell:
                                agent.waiting_at_cell = best_cell
                                self.waiting_cell_reservations[best_cell] = agent.name
                                print(f"  Agent {agent.name} will go to waiting cell {best_cell}")
                            else:
                                agent.returning_home = True
                                print(f"  Agent {agent.name} returning to home position")
                
                # Move agent
                agent.x = current_state.x
                agent.y = current_state.y
                agent.pitch = current_state.orientation
                agent.path_index += 1
                
                # Check if agent reached waiting cell
                if agent.waiting_at_cell and (agent.x, agent.y) == agent.waiting_at_cell:
                    agents_reached_waiting.append(agent.name)
                
                # Check if agent reached home
                if agent.returning_home and (agent.x, agent.y) == (agent.home_x, agent.home_y):
                    agent.returning_home = False
                    agents_reached_home.append(agent.name)
        
        if agents_reached_waiting:
            print(f"  Agents reached waiting cells: {', '.join(agents_reached_waiting)}")
        if agents_reached_home:
            print(f"  Agents reached home: {', '.join(agents_reached_home)}")
        
        # Trigger replanning if needed
        need_replan = False
        
        # Replan if task picked up (to assign new agents to freed pickup points)
        if task_picked_up:
            need_replan = True
            print(f"Timestep {self.current_timestep}: Replanning - tasks picked up by {', '.join(task_picked_up)}")
        
        # Replan if task completed
        elif task_completed:
            need_replan = True
            print(f"Timestep {self.current_timestep}: Replanning due to task completion")
        
        # Periodic replanning
        elif self.current_timestep - self.last_replan_timestep >= self.replan_period:
            need_replan = True
            print(f"Timestep {self.current_timestep}: Periodic replanning")
        
        if need_replan:
            self.replan()
    
    def replan(self):
        """
        Replan paths for all agents.
        """
        # First, start executing tasks from sequences for agents without current tasks
        for agent in self.agvs.values():
            if not agent.current_task and agent.task_sequence:
                # Get next task from sequence
                task = None
                for t in agent.task_sequence:
                    if not t.is_executing and not t.is_completed:
                        task = t
                        break
                
                if task:
                    agent.task_sequence.remove(task)
                    agent.current_task = task
                    task.is_executing = True
                    task.assigned_to = agent.name
                    # Clear waiting or returning status if agent gets a new task
                    if agent.waiting_at_cell:
                        if agent.waiting_at_cell in self.waiting_cell_reservations:
                            if self.waiting_cell_reservations[agent.waiting_at_cell] == agent.name:
                                del self.waiting_cell_reservations[agent.waiting_at_cell]
                        agent.waiting_at_cell = None
                    if agent.returning_home:
                        agent.returning_home = False
                    print(f"  Agent {agent.name} starting task {task.task_id}")
        
        # Clear stale waiting reservations
        self.clear_stale_waiting_reservations()
        
        # Reassign tasks to free agents (including those at waiting cells or returning home)
        self.dynamic_task_assignment_lns()
        
        # Start newly assigned tasks
        for agent in self.agvs.values():
            if not agent.current_task and agent.task_sequence:
                task = None
                for t in agent.task_sequence:
                    if not t.is_executing and not t.is_completed:
                        task = t
                        break
                
                if task:
                    agent.task_sequence.remove(task)
                    agent.current_task = task
                    task.is_executing = True
                    task.assigned_to = agent.name
                    print(f"  Agent {agent.name} starting task {task.task_id}")
        
        # Plan paths using windowed PBS
        print(f"  Planning paths with window={self.window}")
        solution = self.pbs_high_level()
        
        # Update agent paths
        for agent_name, path in solution.items():
            agent = self.agvs[agent_name]
            agent.path = path
            agent.path_index = 0
        
        self.last_replan_timestep = self.current_timestep
        
        # Report statistics
        active_agents = sum(1 for a in self.agvs.values() if a.current_task)
        waiting_agents = sum(1 for a in self.agvs.values() if a.waiting_at_cell and not a.current_task)
        returning_agents = sum(1 for a in self.agvs.values() if a.returning_home)
        idle_agents = len(self.agvs) - active_agents - waiting_agents - returning_agents
        remaining_tasks = sum(1 for t in self.tasks if not t.is_completed)
        
        print(f"  Agents: {active_agents} active, {waiting_agents} waiting, {returning_agents} returning, {idle_agents} idle")
        print(f"  Tasks: {len(self.tasks) - remaining_tasks} completed, {remaining_tasks} remaining")
    
    def generate_trajectory(self):
        """
        Generate complete trajectory using LNS-wPBS.
        """
        print("="*60)
        print("Running LNS-wPBS algorithm...")
        print(f"Parameters: window={self.window}, replan_period={self.replan_period}")
        print("="*60)
        
        # Initial planning
        self.replan()
        
        trajectory_data = []
        max_timesteps = 1000  # Maximum simulation time
        
        # Run simulation
        for t in range(max_timesteps):
            # Record current state for each agent
            for agent_name, agent in self.agvs.items():
                emergency = False
                destination = ""
                task_id = ""
                
                # Get current task information
                if agent.current_task:
                    emergency = agent.current_task.priority == "Urgent"
                    if agent.loaded:
                        destination = agent.current_task.end_point
                
                # Check if agent is performing pickup/dropoff THIS timestep
                if agent.last_action in [Action.PICKUP, Action.DROPOFF]:
                    task_id = agent.last_action_task if agent.last_action_task else ""
                
                trajectory_data.append({
                    'timestamp': t,
                    'name': agent_name,
                    'X': agent.x,
                    'Y': agent.y,
                    'pitch': agent.pitch,
                    'loaded': str(agent.loaded).lower(),
                    'destination': destination,
                    'Emergency': str(emergency).lower(),
                    'task-id': task_id
                })
            
            # Execute timestep
            self.execute_timestep()
            
            # Check if all tasks completed
            if all(t.is_completed for t in self.tasks):
                # Let agents return home
                all_home = True
                for agent in self.agvs.values():
                    if (agent.x, agent.y) != (agent.home_x, agent.home_y):
                        all_home = False
                        break
                
                if all_home:
                    print(f"\nAll tasks completed and agents returned home at timestep {t}")
                    break
        
        return trajectory_data
    
    def save_trajectory(self, filename: str, trajectory_data: List[Dict]):
        """Save trajectory to CSV file"""
        with open(filename, 'w', newline='') as f:
            fieldnames = ['timestamp', 'name', 'X', 'Y', 'pitch', 'loaded', 
                         'destination', 'Emergency', 'task-id']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Sort by timestamp and name
            sorted_data = sorted(trajectory_data, key=lambda x: (x['timestamp'], x['name']))
            writer.writerows(sorted_data)
    
    def validate_solution(self, trajectory_data: List[Dict]) -> bool:
        """Validate the generated solution for collisions and task completion"""
        print("\nValidating solution...")
        
        # Track task pickups and dropoffs
        task_events = defaultdict(list)  # task_id -> list of (event, agent, timestamp)
        
        for entry in trajectory_data:
            if entry['task-id']:
                task_id = entry['task-id']
                agent = entry['name']
                timestamp = entry['timestamp']
                task_events[task_id].append((agent, timestamp))
        
        # Check for duplicate task executions
        duplicate_tasks = []
        for task_id, events in task_events.items():
            unique_agents = set(agent for agent, _ in events)
            if len(unique_agents) > 1:
                duplicate_tasks.append(task_id)
                print(f"  WARNING: Task {task_id} handled by multiple agents: {unique_agents}")
        
        # Check FIFO ordering at each pickup point
        fifo_violations = []
        for pickup_point, queue in self.task_queues.items():
            pickup_times = {}
            for task in queue:
                if task.task_id in task_events:
                    events = task_events[task.task_id]
                    if events:
                        pickup_times[task.task_id] = min(t for _, t in events)
            
            # Check if tasks were picked up in order
            task_ids_in_queue = [t.task_id for t in queue if t.task_id in pickup_times]
            for i in range(len(task_ids_in_queue) - 1):
                task1, task2 = task_ids_in_queue[i], task_ids_in_queue[i+1]
                if task1 in pickup_times and task2 in pickup_times:
                    if pickup_times[task1] > pickup_times[task2]:
                        fifo_violations.append((pickup_point, task1, task2))
                        print(f"  FIFO violation at {pickup_point}: {task2} picked up before {task1}")
        
        # Check for vertex collisions
        timestep_positions = defaultdict(lambda: defaultdict(list))
        
        for entry in trajectory_data:
            t = entry['timestamp']
            pos = (entry['X'], entry['Y'])
            agent = entry['name']
            timestep_positions[t][pos].append(agent)
        
        vertex_collision_count = 0
        for t, positions in timestep_positions.items():
            for pos, agents in positions.items():
                if len(agents) > 1:
                    vertex_collision_count += 1
                    if vertex_collision_count <= 5:  # Only print first few
                        print(f"  Vertex collision at t={t}, pos={pos}: {agents}")
        
        # Check for edge collisions (swapping)
        edge_collision_count = 0
        agent_positions = defaultdict(dict)  # agent -> time -> (x, y)
        
        for entry in trajectory_data:
            t = entry['timestamp']
            agent = entry['name']
            pos = (entry['X'], entry['Y'])
            agent_positions[agent][t] = pos
        
        # Check each pair of agents for swapping
        agents = list(agent_positions.keys())
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                
                # Check all consecutive timesteps
                max_t = max(max(agent_positions[agent1].keys(), default=0), 
                          max(agent_positions[agent2].keys(), default=0))
                
                for t in range(max_t):
                    if (t in agent_positions[agent1] and t+1 in agent_positions[agent1] and
                        t in agent_positions[agent2] and t+1 in agent_positions[agent2]):
                        
                        pos1_t = agent_positions[agent1][t]
                        pos1_t1 = agent_positions[agent1][t+1]
                        pos2_t = agent_positions[agent2][t]
                        pos2_t1 = agent_positions[agent2][t+1]
                        
                        # Check if they swapped positions
                        if pos1_t == pos2_t1 and pos2_t == pos1_t1:
                            edge_collision_count += 1
                            if edge_collision_count <= 5:
                                print(f"  Edge collision: {agent1} and {agent2} swapped at t={t}")
        
        # Check task completion
        completed_tasks = set(task_events.keys())
        
        print(f"Tasks completed: {len(completed_tasks)}/{len(self.tasks)}")
        print(f"Duplicate task executions: {len(duplicate_tasks)}")
        print(f"FIFO violations: {len(fifo_violations)}")
        print(f"Vertex collisions detected: {vertex_collision_count}")
        print(f"Edge collisions detected: {edge_collision_count}")
        
        total_collisions = vertex_collision_count + edge_collision_count
        return (total_collisions == 0 and len(completed_tasks) == len(self.tasks) and 
                len(duplicate_tasks) == 0 and len(fifo_violations) == 0)

def main():
    """Main execution function"""
    # Initialize solver with windowing parameters
    solver = LNSwPBSSolver(window=10, replan_period=5)
    
    # Get current directory
    current_dir = os.getcwd()
    
    # Load data files
    map_file = os.path.join(current_dir, 'input/map_data_full.csv')
    task_file = os.path.join(current_dir, 'input/task_csv_full.csv')
    output_file = os.path.join(current_dir, 'output/agv_trajectory.csv')
    
    # Check if input files exist
    if not os.path.exists(map_file):
        print(f"Error: {map_file} not found!")
        return
    if not os.path.exists(task_file):
        print(f"Error: {task_file} not found!")
        return
    
    print("="*60)
    print("LNS-wPBS MAPD Solver")
    print("="*60)
    print(f"Loading data from {map_file} and {task_file}...")
    
    solver.load_data(map_file, task_file)
    
    print(f"Loaded {len(solver.agvs)} AGVs and {len(solver.tasks)} tasks")
    print(f"Start points: {list(solver.start_points.keys())}")
    print(f"End points: {list(solver.end_points.keys())}")
    print(f"AGVs: {list(solver.agvs.keys())}")
    
    print("\nGenerating trajectory with LNS-wPBS algorithm...")
    start_time = time.time()
    
    # Generate trajectory
    trajectory = solver.generate_trajectory()
    
    elapsed_time = time.time() - start_time
    print(f"\nPlanning completed in {elapsed_time:.2f} seconds")
    
    # Save to file
    print(f"Saving trajectory to {output_file}...")
    solver.save_trajectory(output_file, trajectory)
    
    # Calculate statistics
    if trajectory:
        max_time = max(t['timestamp'] for t in trajectory)
        tasks_completed = len(set(t['task-id'] for t in trajectory if t['task-id']))
        
        print("\n" + "="*60)
        print("Results:")
        print(f"  Makespan: {max_time} timesteps")
        print(f"  Tasks completed: {tasks_completed}/{len(solver.tasks)}")
        print(f"  Planning time: {elapsed_time:.2f} seconds")
        print(f"  Output saved to: {output_file}")
        
        # Validate solution
        solver.validate_solution(trajectory)
        print("="*60)
    else:
        print("No trajectory generated!")

if __name__ == "__main__":
    main()