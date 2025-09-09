import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
from collections import defaultdict
import os

class AGVVisualizer:
    def __init__(self, map_file='map_data.csv', task_file='task_csv.csv', 
                 trajectory_file='agv_trajectory.csv'):
        """Initialize the AGV Visualizer with input files."""
        self.map_file = map_file
        self.task_file = task_file
        self.trajectory_file = trajectory_file
        
        # Warehouse dimensions
        self.width = 20
        self.height = 20
        
        # Data storage
        self.start_points = {}
        self.end_points = {}
        self.agvs_initial = {}
        self.tasks = []
        self.trajectories = defaultdict(list)
        self.agv_tasks = {}  # Track which task each AGV is carrying
        self.tasks_remaining = defaultdict(lambda: {'pickup': defaultdict(int), 'dropoff': defaultdict(int)})
        
        # Colors for visualization
        self.colors = {
            'empty_agv': '#4CAF50',  # Green for empty AGV
            'loaded_agv': '#FF5722',  # Red for loaded AGV
            'start_point': '#2196F3',  # Blue for pick-up points
            'end_point': '#FFC107',  # Yellow for drop-off points
            'pickup_cell': '#E3F2FD',  # Light blue for pick-up cells
            'dropoff_cell': '#FFF9C4',  # Light yellow for drop-off cells
            'grid': '#9E9E9E',  # Gray for grid
            'wall': '#424242'  # Dark gray for walls
        }
        
    def load_data(self):
        """Load all CSV files."""
        print("Loading map data...")
        self.load_map_data()
        print("Loading task data...")
        self.load_task_data()
        print("Loading trajectory data...")
        self.load_trajectory_data()
        
    def load_map_data(self):
        """Load map data including start/end points and initial AGV positions."""
        df = pd.read_csv(self.map_file)
        
        for _, row in df.iterrows():
            if row['type'] == 'start_point':
                self.start_points[row['name']] = (int(row['x']), int(row['y']))
            elif row['type'] == 'end_point':
                self.end_points[row['name']] = (int(row['x']), int(row['y']))
            elif row['type'] == 'agv':
                self.agvs_initial[row['name']] = {
                    'x': int(row['x']),
                    'y': int(row['y']),
                    'pitch': int(row['pitch'])
                }
                
    def load_task_data(self):
        """Load task data."""
        df = pd.read_csv(self.task_file)
        self.tasks = df.to_dict('records')
        
    def load_trajectory_data(self):
        """Load AGV trajectory data and track tasks."""
        df = pd.read_csv(self.trajectory_file)
        
        # Group by timestamp
        self.timestamps = sorted(df['timestamp'].unique())
        
        # Initialize task counts from task data
        initial_pickup_counts = defaultdict(int)
        initial_dropoff_counts = defaultdict(int)
        task_to_start = {}  # Map task_id to start_point
        task_to_end = {}    # Map task_id to end_point
        
        for task in self.tasks:
            task_id = task['task_id']
            start = task['start_point']
            end = task['end_point']
            initial_pickup_counts[start] += 1
            initial_dropoff_counts[end] += 1
            task_to_start[task_id] = start
            task_to_end[task_id] = end
        
        # Track completed tasks
        picked_up_tasks = set()
        dropped_off_tasks = set()
        
        for timestamp in self.timestamps:
            frame_data = df[df['timestamp'] == timestamp]
            frame_agvs = []
            
            # Calculate current remaining tasks
            pickup_remaining = initial_pickup_counts.copy()
            dropoff_remaining = initial_dropoff_counts.copy()
            
            for _, row in frame_data.iterrows():
                agv_name = row['name']
                task_id = row.get('task-id', '')
                loaded = row['loaded'] in ['true', 'True', True, 1]
                
                # Update task tracking
                if task_id and task_id not in ['', 'nan', None, 'None'] and str(task_id) != 'nan':
                    if loaded:  # Pick-up action
                        self.agv_tasks[agv_name] = task_id
                        picked_up_tasks.add(task_id)
                    else:  # Drop-off action
                        if agv_name in self.agv_tasks:
                            del self.agv_tasks[agv_name]
                        dropped_off_tasks.add(task_id)
                
                # Get current task for this AGV
                current_task = self.agv_tasks.get(agv_name, '')
                
                agv_data = {
                    'name': agv_name,
                    'x': int(row['X']),
                    'y': int(row['Y']),
                    'pitch': int(row['pitch']),
                    'loaded': loaded,
                    'destination': row.get('destination', ''),
                    'emergency': row.get('Emergency', False) in ['true', 'True', True, 1],
                    'task_id': current_task  # Use tracked task instead of just the current row value
                }
                frame_agvs.append(agv_data)
            
            # Update remaining counts based on completed tasks
            for task_id in picked_up_tasks:
                if task_id in task_to_start:
                    start = task_to_start[task_id]
                    if start in pickup_remaining:
                        pickup_remaining[start] -= 1
            
            for task_id in dropped_off_tasks:
                if task_id in task_to_end:
                    end = task_to_end[task_id]
                    if end in dropoff_remaining:
                        dropoff_remaining[end] -= 1
            
            # Store remaining tasks for this timestamp
            self.tasks_remaining[timestamp]['pickup'] = pickup_remaining
            self.tasks_remaining[timestamp]['dropoff'] = dropoff_remaining
            
            self.trajectories[timestamp] = frame_agvs
    
    def get_pickup_cells(self):
        """Calculate actual pick-up cells adjacent to loading ramps."""
        pickup_cells = []
        for name, (x, y) in self.start_points.items():
            # Determine adjacent interior cell based on ramp position
            if x == 1:  # Left wall
                pickup_cells.append((x + 1, y))
            elif x == 20:  # Right wall
                pickup_cells.append((x - 1, y))
            elif y == 1:  # Bottom wall
                pickup_cells.append((x, y + 1))
            elif y == 20:  # Top wall
                pickup_cells.append((x, y - 1))
        return pickup_cells
    
    def get_dropoff_cells(self):
        """Calculate possible drop-off cells around offload ramps."""
        dropoff_cells = []
        for name, (x, y) in self.end_points.items():
            # Add all four orthogonally adjacent cells
            adjacent = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for ax, ay in adjacent:
                if 1 <= ax <= 20 and 1 <= ay <= 20:  # Within warehouse bounds
                    dropoff_cells.append((ax, ay))
        return dropoff_cells
            
    def draw_agv(self, ax, x, y, pitch, loaded, name, scale=0.3):
        """Draw an AGV as a triangle pointing in the direction of movement."""
        # Convert pitch to radians
        angle_rad = np.radians(pitch)
        
        # Define triangle vertices (pointing east when pitch=0)
        triangle = np.array([
            [0.5, 0],      # Tip
            [-0.3, 0.3],   # Upper back
            [-0.3, -0.3]   # Lower back
        ]) * scale
        
        # Rotate triangle based on pitch
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        rotated_triangle = triangle @ rotation_matrix.T
        
        # Translate to position
        rotated_triangle[:, 0] += x
        rotated_triangle[:, 1] += y
        
        # Choose color based on loaded status
        color = self.colors['loaded_agv'] if loaded else self.colors['empty_agv']
        
        # Draw the triangle
        triangle_patch = patches.Polygon(rotated_triangle, closed=True, 
                                       facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(triangle_patch)
        
        # Add AGV name
        ax.text(x, y-0.6, name, fontsize=6, ha='center', va='top')
        
    def create_frame(self, timestamp):
        """Create a single frame of the visualization."""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Set up the plot with correct coordinate system
        # Grid cells go from 1 to 20, but we need extra space for display
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0.5, 20.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert Y-axis so (1,1) is at bottom-left visually
        
        # Draw grid cells (outlines)
        for x in range(1, self.width + 1):
            for y in range(1, self.height + 1):
                rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                        linewidth=0.5, edgecolor=self.colors['grid'], 
                                        facecolor='white', alpha=1.0)
                ax.add_patch(rect)
        
        # Highlight pick-up cells
        for px, py in self.get_pickup_cells():
            rect = patches.Rectangle((px-0.5, py-0.5), 1, 1, 
                                    facecolor=self.colors['pickup_cell'], 
                                    edgecolor=self.colors['grid'], linewidth=0.5, alpha=0.5)
            ax.add_patch(rect)
            
        # Highlight drop-off cells
        for dx, dy in self.get_dropoff_cells():
            rect = patches.Rectangle((dx-0.5, dy-0.5), 1, 1, 
                                    facecolor=self.colors['dropoff_cell'], 
                                    edgecolor=self.colors['grid'], linewidth=0.5, alpha=0.3)
            ax.add_patch(rect)
            
        # Draw warehouse boundary
        rect = patches.Rectangle((0.5, 0.5), self.width, self.height, 
                                linewidth=3, edgecolor=self.colors['wall'], facecolor='none')
        ax.add_patch(rect)
        
        # Draw start points (pick-up locations - on the walls) with task counts
        pickup_counts = self.tasks_remaining[timestamp]['pickup'] if timestamp in self.tasks_remaining else {}
        for name, (x, y) in self.start_points.items():
            circle = patches.Circle((x, y), 0.25, color=self.colors['start_point'], 
                                   alpha=0.9, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, f"P:{name[:2]}", fontsize=5, ha='center', va='center', 
                   color='white', weight='bold', zorder=6)
            # Add task count
            count = pickup_counts.get(name, 0)
            if count > 0:
                ax.text(x, y-0.4, f"[{count}]", fontsize=6, ha='center', va='top', 
                       color='blue', weight='bold', zorder=6)
            
        # Draw end points (drop-off locations - on the walls) with task counts
        dropoff_counts = self.tasks_remaining[timestamp]['dropoff'] if timestamp in self.tasks_remaining else {}
        for name, (x, y) in self.end_points.items():
            square = patches.Rectangle((x-0.25, y-0.25), 0.5, 0.5, 
                                      color=self.colors['end_point'], alpha=0.9, zorder=5)
            ax.add_patch(square)
            ax.text(x, y, f"D:{name[:2]}", fontsize=5, ha='center', va='center', 
                   color='black', weight='bold', zorder=6)
            # Add task count
            count = dropoff_counts.get(name, 0)
            if count > 0:
                ax.text(x, y-0.4, f"[{count}]", fontsize=6, ha='center', va='top', 
                       color='orange', weight='bold', zorder=6)
            
        # Draw AGVs at current timestamp
        if timestamp in self.trajectories:
            for agv in self.trajectories[timestamp]:
                self.draw_agv(ax, agv['x'], agv['y'], agv['pitch'], 
                            agv['loaded'], agv['name'])
                
                # Show task ID if AGV is carrying a task
                if agv.get('task_id') and agv['task_id'] not in ['', 'nan', None]:
                    ax.text(agv['x'], agv['y']+0.7, f"Task: {agv['task_id']}", 
                           fontsize=5, ha='center', va='bottom', 
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # Add title and timestamp
        ax.set_title(f'AGV Warehouse Simulation - Timestep: {timestamp}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=10)
        ax.set_ylabel('Y Coordinate', fontsize=10)
        
        # Add legend
        legend_elements = [
            patches.Patch(color=self.colors['empty_agv'], label='Empty AGV'),
            patches.Patch(color=self.colors['loaded_agv'], label='Loaded AGV'),
            patches.Circle((0, 0), 0.1, color=self.colors['start_point'], label='Loading Ramp'),
            patches.Rectangle((0, 0), 0.1, 0.1, color=self.colors['end_point'], label='Offload Ramp'),
            patches.Patch(color=self.colors['pickup_cell'], label='Pick-up Cell'),
            patches.Patch(color=self.colors['dropoff_cell'], label='Drop-off Cell')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Add grid labels
        ax.set_xticks(range(1, self.width + 1))
        ax.set_yticks(range(1, self.height + 1))
        ax.set_xticklabels(range(1, self.width + 1))
        ax.set_yticklabels(range(self.height, 0, -1))  # Reverse Y labels for correct display
        
        plt.tight_layout()
        return fig, ax
        
    def create_animation(self, output_file='agv_simulation.mp4', fps=2):
        """Create animated MP4 of the entire simulation."""
        print(f"Creating animation with {len(self.timestamps)} frames...")
        
        # Create figure for animation
        fig, ax = plt.subplots(figsize=(12, 12))
        
        def init():
            ax.clear()
            return []
        
        def animate(frame_idx):
            ax.clear()
            timestamp = self.timestamps[frame_idx]
            
            # Set up the plot with correct coordinate system
            ax.set_xlim(0.5, 20.5)
            ax.set_ylim(0.5, 20.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            # Draw grid cells (outlines)
            for x in range(1, self.width + 1):
                for y in range(1, self.height + 1):
                    rect = patches.Rectangle((x-0.5, y-0.5), 1, 1, 
                                            linewidth=0.5, edgecolor=self.colors['grid'], 
                                            facecolor='white', alpha=1.0)
                    ax.add_patch(rect)
            
            # Highlight pick-up cells
            for px, py in self.get_pickup_cells():
                rect = patches.Rectangle((px-0.5, py-0.5), 1, 1, 
                                        facecolor=self.colors['pickup_cell'], 
                                        edgecolor=self.colors['grid'], linewidth=0.5, alpha=0.5)
                ax.add_patch(rect)
                
            # Highlight drop-off cells
            for dx, dy in self.get_dropoff_cells():
                rect = patches.Rectangle((dx-0.5, dy-0.5), 1, 1, 
                                        facecolor=self.colors['dropoff_cell'], 
                                        edgecolor=self.colors['grid'], linewidth=0.5, alpha=0.3)
                ax.add_patch(rect)
                
            # Draw warehouse boundary
            rect = patches.Rectangle((0.5, 0.5), self.width, self.height, 
                                    linewidth=3, edgecolor=self.colors['wall'], facecolor='none')
            ax.add_patch(rect)
            
            # Draw start points with remaining task counts
            pickup_counts = self.tasks_remaining[timestamp]['pickup'] if timestamp in self.tasks_remaining else {}
            for name, (x, y) in self.start_points.items():
                circle = patches.Circle((x, y), 0.25, color=self.colors['start_point'], 
                                       alpha=0.9, zorder=5)
                ax.add_patch(circle)
                ax.text(x, y, f"P:{name[:2]}", fontsize=5, ha='center', va='center', 
                       color='white', weight='bold', zorder=6)
                # Add task count
                count = pickup_counts.get(name, 0)
                if count > 0:
                    ax.text(x, y-0.4, f"[{count}]", fontsize=6, ha='center', va='top', 
                           color='blue', weight='bold', zorder=6)
                
            # Draw end points with remaining task counts
            dropoff_counts = self.tasks_remaining[timestamp]['dropoff'] if timestamp in self.tasks_remaining else {}
            for name, (x, y) in self.end_points.items():
                square = patches.Rectangle((x-0.25, y-0.25), 0.5, 0.5, 
                                          color=self.colors['end_point'], alpha=0.9, zorder=5)
                ax.add_patch(square)
                ax.text(x, y, f"D:{name[:2]}", fontsize=5, ha='center', va='center', 
                       color='black', weight='bold', zorder=6)
                # Add task count
                count = dropoff_counts.get(name, 0)
                if count > 0:
                    ax.text(x, y-0.4, f"[{count}]", fontsize=6, ha='center', va='top', 
                           color='orange', weight='bold', zorder=6)
                
            # Draw AGVs
            if timestamp in self.trajectories:
                for agv in self.trajectories[timestamp]:
                    self.draw_agv(ax, agv['x'], agv['y'], agv['pitch'], 
                                agv['loaded'], agv['name'])
                    
                    # Show task ID if AGV is carrying a task
                    if agv.get('task_id') and agv['task_id'] not in ['', 'nan', None]:
                        ax.text(agv['x'], agv['y']+0.7, f"Task: {agv['task_id']}", 
                               fontsize=5, ha='center', va='bottom', 
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Add title and timestamp
            ax.set_title(f'AGV Warehouse Simulation - Timestep: {timestamp}', fontsize=14, fontweight='bold')
            ax.set_xlabel('X Coordinate', fontsize=10)
            ax.set_ylabel('Y Coordinate', fontsize=10)
            
            # Add legend
            legend_elements = [
                patches.Patch(color=self.colors['empty_agv'], label='Empty AGV'),
                patches.Patch(color=self.colors['loaded_agv'], label='Loaded AGV'),
                patches.Circle((0, 0), 0.1, color=self.colors['start_point'], label='Loading Ramp'),
                patches.Rectangle((0, 0), 0.1, 0.1, color=self.colors['end_point'], label='Offload Ramp'),
                patches.Patch(color=self.colors['pickup_cell'], label='Pick-up Cell'),
                patches.Patch(color=self.colors['dropoff_cell'], label='Drop-off Cell')
            ]
            ax.legend(handles=legend_elements, loc='upper left')
            
            # Add grid labels
            ax.set_xticks(range(1, self.width + 1))
            ax.set_yticks(range(1, self.height + 1))
            ax.set_xticklabels(range(1, self.width + 1))
            ax.set_yticklabels(range(self.height, 0, -1))  # Reverse Y labels
            
            # Progress indicator
            if frame_idx % 10 == 0:
                print(f"Processing frame {frame_idx+1}/{len(self.timestamps)}")
            
            return []
        
        # Create animation
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(self.timestamps), interval=1000/fps, blit=False)
        
        # Save as MP4
        print(f"Saving animation to {output_file}...")
        try:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='AGV Simulator'), bitrate=1800)
            anim.save(output_file, writer=writer)
            print(f"Animation saved successfully as '{output_file}'")
        except:
            print("FFmpeg not found. Please install FFmpeg or trying saving as GIF instead.")
            raise
        
        plt.close()
        
        print(f"Total frames: {len(self.timestamps)}")
        print(f"Animation duration: {len(self.timestamps)/fps:.1f} seconds at {fps} fps")
        
    def run(self, output_file='agv_simulation.mp4', fps=2):
        """Main method to run the visualization."""
        try:
            # Load all data
            self.load_data()
            
            # Print summary
            print("\n=== Data Summary ===")
            print(f"Number of pick-up points: {len(self.start_points)}")
            print(f"Number of drop-off points: {len(self.end_points)}")
            print(f"Number of AGVs: {len(self.agvs_initial)}")
            print(f"Number of tasks: {len(self.tasks)}")
            print(f"Simulation duration: {len(self.timestamps)} timesteps")
            
            # Create animation
            self.create_animation(output_file, fps)
            
        except FileNotFoundError as e:
            print(f"Error: Could not find input file - {e}")
            print("  - input/map_data.csv")
            print("  - input/task_csv.csv")
            print("  - output/agv_trajectory.csv")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run the AGV visualizer."""
    # Create visualizer instance
    visualizer = AGVVisualizer(
        map_file='input/map_data.csv',
        task_file='input/task_csv.csv',
        trajectory_file='output/agv_trajectory.csv'
    )
    
    # Run visualization and create MP4
    # You can adjust fps (frames per second) for faster/slower playback
    visualizer.run(output_file='output/agv_simulation.mp4', fps=30)

if __name__ == "__main__":
    main()