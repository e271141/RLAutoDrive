from airsim_client import *
from ac_model import ACModel
import time
import numpy as np
import threading
import json
import os
import glob
import datetime

MAX_FRAME_COUNT = 155	# The largest number of minibatches that the RAM won't crash

# A class represents a local agent that drive the vehicle, train & update the model
# The basic structure of the class was inherited from DistributedAgent -> distributed_agent.py.
class LocalAgent():
    def __init__(self, parameters):
        required_parameters = ['data_dir', 'max_epoch_runtime_sec', 'replay_memory_size', \
        						'batch_size', 'min_epsilon', 'per_iter_epsilon_reduction', \
        						'experiment_name', 'train_conv_layers']

        for required_parameter in required_parameters:
            if required_parameter not in parameters:
                raise ValueError('Missing required parameter {0}'.format(required_parameter))

        parameters['role_type'] = 'agent'

        
        print('Starting time: {0}'.format(datetime.datetime.utcnow()), file=sys.stderr)
        self.__model_buffer = None
        self.__model = None
        self.__airsim_started = False
        self.__data_dir = parameters['data_dir']
        self.__per_iter_epsilon_reduction = float(parameters['per_iter_epsilon_reduction'])
        self.__min_epsilon = float(parameters['min_epsilon'])
        self.__max_epoch_runtime_sec = float(parameters['max_epoch_runtime_sec'])
        self.__replay_memory_size = int(parameters['replay_memory_size'])
        self.__batch_size = int(parameters['batch_size'])
        self.__experiment_name = parameters['experiment_name']
        self.__train_conv_layers = bool((parameters['train_conv_layers'].lower().strip() == 'true'))
        self.__epsilon = 1
        self.__num_batches_run = 0
        self.__last_checkpoint_batch_count = 0
        
        if 'batch_update_frequency' in parameters:
            self.__batch_update_frequency = int(parameters['batch_update_frequency'])
        
        if 'weights_path' in parameters:
            self.__weights_path = parameters['weights_path']
        else:
            self.__weights_path = None
            
        if 'airsim_path' in parameters:
            self.__airsim_path = parameters['airsim_path']
        else:
            self.__airsim_path = None

        self.__car_client = None
        self.__car_controls = None

        self.__minibatch_dir = os.path.join(self.__data_dir, 'minibatches')
        self.__output_model_dir = os.path.join(self.__data_dir, 'models')

        self.__make_dir_if_not_exist(self.__minibatch_dir)
        self.__make_dir_if_not_exist(self.__output_model_dir)
        self.__last_model_file = ''

        self.__possible_ip_addresses = []
        self.__trainer_ip_address = None

        self.__experiences = {}

        self.__init_road_points()
        self.__init_reward_points()

    # Starts the agent
    def start(self):
        self.__run_function()

    def __run_function(self):
    	print("Starting run function")

    	self.__model = ACModel(self.__weights_path, self.__train_conv_layers)

    	# Connect to AirSim exe
    	self.__connect_to_airsim()

    	# Fill the replay memory by driving randomly.
    	# If the first epoch didn't generate enough data,
    	# the training can still carry on
    	self.__fill_replay_memory()

    	print('Replay memory filled. Starting main loop...')
        while True:
            try:
                if (self.__model is not None):

                    #Generate a series of training examples by driving the vehicle in AirSim
                    print('Running Airsim Epoch.')
                    experiences, frame_count = self.__run_epoch(False)

                    # If we didn't immediately crash, train on the gathered experiences
                    if (frame_count > 0):
                        print('Generating {0} minibatches...'.format(frame_count))

                        print('Sampling Experiences.')
                        # Sample experiences from the replay memory
                        sampled_experiences = self.__sample_experiences(experiences, frame_count, True)

                        self.__num_batches_run += frame_count
                        
                        # If we successfully sampled, train on the collected minibatches and send the gradients to the trainer node
                        if (len(sampled_experiences) > 0):
                            print('Publishing AirSim Epoch.')
                            self.__publish_batch_and_update_model(sampled_experiences, frame_count)
            
            # Occasionally, the AirSim exe will stop working.
            # For example, if a user connects to the node to visualize progress.
            # In that case, attempt to reconnect.
            except msgpackrpc.error.TimeoutError:
                print('Lost connection to AirSim. Attempting to reconnect.')
                self.__connect_to_airsim()



    def __fill_replay_memory(self):
        while True:
            try:
            	percent_full = 100.0 * len(self.__experiences['actions'])/self.__replay_memory_size
            	print('Replay memory now contains {0} members. ({1}% full)'\
            				.format(len(self.__experiences['actions']), percent_full))
            	if percent_full >= 100.00:
            		break
            	self.__run_epoch(True)
            
            except msgpackrpc.error.TimeoutError:
                print('Lost connection to AirSim. Attempting to reconnect.')
                self.__connect_to_airsim() 

    # Sample experiences from the replay memory
    def __sample_experiences(self, experiences, frame_count, sample_randomly):

        sampled_experiences = {}
        sampled_experiences['pre_states'] = []
        sampled_experiences['post_states'] = []
        sampled_experiences['actions'] = []
        sampled_experiences['rewards'] = []
        sampled_experiences['predicted_rewards'] = []
        sampled_experiences['is_not_terminal'] = []
        sampled_experiences['speed'] = []

        # Compute the surprise factor, which is the difference between the predicted an the actual Q value for each state.
        # We can use that to weight examples so that we are more likely to train on examples that the model got wrong.
        suprise_factor = np.abs(np.array(experiences['rewards'], dtype=np.dtype(float)) \
        				- np.array(experiences['predicted_rewards'], dtype=np.dtype(float)))
        suprise_factor_normalizer = np.sum(suprise_factor)
        suprise_factor /= float(suprise_factor_normalizer)

        # Generate one minibatch for each frame of the run
        ratio = 1
        if frame_count > MAX_FRAME_COUNT:
            ratio = math.ceil(float(frame_count)/MAX_FRAME_COUNT)
        for _ in range(0, frame_count, ratio):

            if sample_randomly:
                idx_set = set(np.random.choice(list(range(0, suprise_factor.shape[0], 1)), \
                								size=(self.__batch_size), replace=False))
            else:
                idx_set = set(np.random.choice(list(range(0, suprise_factor.shape[0], 1)), \
                					size=(self.__batch_size), replace=False, p=suprise_factor))
        
            sampled_experiences['pre_states'] += [experiences['pre_states'][i] for i in idx_set]
            sampled_experiences['post_states'] += [experiences['post_states'][i] for i in idx_set]
            sampled_experiences['actions'] += [experiences['actions'][i] for i in idx_set]
            sampled_experiences['rewards'] += [experiences['rewards'][i] for i in idx_set]
            sampled_experiences['predicted_rewards'] += [experiences['predicted_rewards'][i] for i in idx_set]
            sampled_experiences['is_not_terminal'] += [experiences['is_not_terminal'][i] for i in idx_set]
            sampled_experiences['speed'] += [experiences['speed'][i] for i in idx_set]
            
        return sampled_experiences

    # Connects to the AirSim Exe.
    # Assume that it is already running. 
    # After 10 successive attempts, attempt to restart the executable.
    def __connect_to_airsim(self):
        attempt_count = 0
        while True:
            try:
                print('Attempting to connect to AirSim (attempt {0})'.format(attempt_count))
                self.__car_client = CarClient()
                self.__car_client.confirmConnection()
                self.__car_client.enableApiControl(True)
                self.__car_controls = CarControls()
                print('Connected!')
                return
            except:
                print('Failed to connect.')
                attempt_count += 1
                if (attempt_count % 10 == 0):
                    print('10 consecutive failures to connect. \
                    			Attempting to start AirSim on my own.')
                    
                os.system('START "" powershell.exe {0}'.format(os.path.join(self.__airsim_path, 
                					'AD_Cookbook_Start_AirSim.ps1 neighborhood -windowed')))
                print('Waiting a few seconds.')
                time.sleep(10)

    # Runs an interation of data generation from AirSim.
    # Data will be saved in the replay memory.
    def __run_epoch(self, always_random):
        print('Running AirSim epoch.')
        
        # Pick a random starting point on the roads
        starting_points, starting_direction = self.__get_next_starting_point()
        
        # Initialize the state buffer.
        # For now, save 4 images at 0.01 second intervals.
        state_buffer_len = 4
        state_buffer = []
        wait_delta_sec = 0.01

        print('Getting Pose')
        self.__car_client.simSetPose(Pose(Vector3r(starting_points[0], starting_points[1], starting_points[2]), AirSimClientBase.toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)

        # Currently, simSetPose does not allow us to set the velocity. 
        # So, if we crash and call simSetPose, the car will be still moving at its previous velocity.
        # We need the car to stop moving, so push the brake and wait for a few seconds.
        print('Waiting for momentum to die')
        self.__set_vehicle_control(0,0,1)
        time.sleep(4)
        
        print('Resetting')
        self.__car_client.simSetPose(Pose(Vector3r(starting_points[0], starting_points[1], starting_points[2]), AirSimClientBase.toQuaternion(starting_direction[0], starting_direction[1], starting_direction[2])), True)

        #Start the car rolling so it doesn't get stuck
        print('Running car for a few seconds...')
        self.__set_vehicle_control(0,1,0)
        
        # While the car is rolling, start initializing the state buffer
        stop_run_time =datetime.datetime.now() + datetime.timedelta(seconds=2)
        while(datetime.datetime.now() < stop_run_time):
            time.sleep(wait_delta_sec)
            state_buffer = self.__append_to_ring_buffer(self.__get_image(), state_buffer, state_buffer_len)
        done = False
        actions = [] #records the state we go to
        pre_states = []
        post_states = []
        rewards = []
        predicted_rewards = []
        speed = []
        car_state = self.__car_client.getCarState()

        start_time = datetime.datetime.utcnow()
        end_time = start_time + datetime.timedelta(seconds=self.__max_epoch_runtime_sec)
        
        num_random = 0
        far_off = False
        
        # Main data collection loop
        while not done:
            collision_info = self.__car_client.getCollisionInfo()
            utc_now = datetime.datetime.utcnow()
            
            # Check for terminal conditions:
            # 1) Car has collided
            # 2) Car is stopped
            # 3) The run has been running for longer than max_epoch_runtime_sec. 
            #       This constraint is so the model doesn't end up having to churn through huge chunks of data, slowing down training
            # 4) The car has run off the road
            if (collision_info.has_collided or car_state.speed < 2 or utc_now > end_time or far_off):
                print('Start time: {0}, end time: {1}'.format(start_time, utc_now), file=sys.stderr)
                if (utc_now > end_time):
                    print('timed out.')
                    print('Full autonomous run finished at {0}'.format(utc_now), file=sys.stderr)
                done = True
                sys.stderr.flush()
            else:

                # The Agent should occasionally pick random action instead of best action
                do_greedy = np.random.random_sample()
                pre_state = copy.deepcopy(state_buffer)
                if (do_greedy < self.__epsilon or always_random):
                    num_random += 1
                    next_state = self.__model.get_random_state()
                    predicted_reward = 0
                    
                else:
                    next_state, predicted_reward = self.__model.predict_state(pre_state)
                    print('Model predicts {0}'.format(next_state))

                # Convert the selected state to a control signal
                next_control_signals = self.__model.state_to_control_signals(next_state, self.__car_client.getCarState())

                # Take the action
                self.__set_vehicle_control(next_control_signals[0], next_control_signals[1], next_control_signals[2])
                
                # Wait for a short period of time to see outcome
                time.sleep(wait_delta_sec)

                # Observe outcome and compute reward from action
                state_buffer = self.__append_to_ring_buffer(self.__get_image(), state_buffer, state_buffer_len)
                car_state = self.__car_client.getCarState()
                collision_info = self.__car_client.getCollisionInfo()
                reward, far_off = self.__compute_reward(collision_info, car_state)
                
                # Add the experience to the set of examples from this iteration
                pre_states.append(pre_state)
                post_states.append(state_buffer)
                rewards.append(reward)
                predicted_rewards.append(predicted_reward)
                actions.append(next_state)
                speed.append(car_state.speed)

        # Only the last state is a terminal state.
        is_not_terminal = [1 for i in range(0, len(actions)-1, 1)]
        is_not_terminal.append(0)
        
        # Add all of the states from this iteration to the replay memory
        self.__add_to_replay_memory('pre_states', pre_states)
        self.__add_to_replay_memory('post_states', post_states)
        self.__add_to_replay_memory('actions', actions)
        self.__add_to_replay_memory('rewards', rewards)
        self.__add_to_replay_memory('predicted_rewards', predicted_rewards)
        self.__add_to_replay_memory('is_not_terminal', is_not_terminal)
        self.__add_to_replay_memory('speed', speed)

        print('Percent random actions: {0}'.format(num_random / max(1, len(actions))))
        print('Num total actions: {0}'.format(len(actions)))
        
        # If we are in the main loop, reduce the epsilon parameter so that the model will be called more often
        # Note: this will be overwritten by the trainer's epsilon if running in distributed mode
        if not always_random:
            self.__epsilon -= self.__per_iter_epsilon_reduction
            self.__epsilon = max(self.__epsilon, self.__min_epsilon)
        
        return self.__experiences, len(actions)

    # Train the model on minibatches and post to the trainer node.
    # The trainer node will respond with the latest version of the model that will be used in further data generation iterations.
    def __publish_batch_and_update_model(self, batches, batches_count):
        # Train and get the gradients
        print('Publishing epoch data and getting latest model from parameter server...')
        gradients = self.__model.get_gradient_update_from_batches(batches)
        
        # Post the data to the trainer node
        if (self.__num_batches_run > self.__batch_update_frequency + self.__last_checkpoint_batch_count):
            self.__model.update_critic()
            
            checkpoint = {}
            checkpoint['model'] = self.__model.to_packet(get_target=True)
            with open(file_name, 'w') as f:
                print('Checkpointing to {0}'.format(file_name))
                f.write(checkpoint_str)
                

    # Computes the reward functinon based on the car position.
    def __compute_reward(self, collision_info, car_state):
        #Define some constant parameters for the reward function
        THRESH_DIST = 3.5                # The maximum distance from the center of the road to compute the reward function
        DISTANCE_DECAY_RATE = 1.2        # The rate at which the reward decays for the distance function
        CENTER_SPEED_MULTIPLIER = 2.0    # The ratio at which we prefer the distance reward to the speed reward

        # If the car has collided, the reward is always zero
        if (collision_info.has_collided):
            return 0.0, True
        
        # If the car is stopped, the reward is always zero
        speed = car_state.speed
        if (speed < 2):
            return 0.0, True
        
        #Get the car position
        position_key = bytes('position', encoding='utf8')
        x_val_key = bytes('x_val', encoding='utf8')
        y_val_key = bytes('y_val', encoding='utf8')

        car_point = np.array([car_state.kinematics_true[position_key][x_val_key], car_state.kinematics_true[position_key][y_val_key], 0])
        
        # Distance component is exponential distance to nearest line
        distance = 999
        
        #Compute the distance to the nearest center line
        for line in self.__reward_points:
            local_distance = 0
            length_squared = ((line[0][0]-line[1][0])**2) + ((line[0][1]-line[1][1])**2)
            if (length_squared != 0):
                t = max(0, min(1, np.dot(car_point-line[0], line[1]-line[0]) / length_squared))
                proj = line[0] + (t * (line[1]-line[0]))
                local_distance = np.linalg.norm(proj - car_point)
            
            distance = min(local_distance, distance)
            
        distance_reward = math.exp(-(distance * DISTANCE_DECAY_RATE))
        
        return distance_reward, distance > THRESH_DIST

    # Initializes the points used for determining the starting point of the vehicle
    def __init_road_points(self):
        self.__road_points = []
        car_start_coords = [0, 0, 0]
        with open(os.path.join(os.path.join(self.__data_dir, 'data'), 'road_lines.txt'), 'r') as f:
            for line in f:
                points = line.split('\t')
                first_point = np.array([float(p) for p in points[0].split(',')] + [0])
                second_point = np.array([float(p) for p in points[1].split(',')] + [0])
                self.__road_points.append(tuple((first_point, second_point)))

        # Points in road_points.txt are in unreal coordinates
        # But car start coordinates are not the same as unreal coordinates
        for point_pair in self.__road_points:
            for point in point_pair:
                point[0] -= car_start_coords[0]
                point[1] -= car_start_coords[1]
                point[0] /= 100
                point[1] /= 100
              
    # Initializes the points used for determining the optimal position of the vehicle during the reward function
    def __init_reward_points(self):
        self.__reward_points = []
        with open(os.path.join(os.path.join(self.__data_dir, 'data'), 'reward_points.txt'), 'r') as f:
            for line in f:
                point_values = line.split('\t')
                first_point = np.array([float(point_values[0]), float(point_values[1]), 0])
                second_point = np.array([float(point_values[2]), float(point_values[3]), 0])
                self.__reward_points.append(tuple((first_point, second_point)))

    # Randomly selects a starting point on the road
    # Used for initializing an iteration of data generation from AirSim
    def __get_next_starting_point(self):
    
        # Get the current state of the vehicle
        car_state = self.__car_client.getCarState()

        # Pick a random road.
        random_line_index = np.random.randint(0, high=len(self.__road_points))
        
        # Pick a random position on the road. 
        # Do not start too close to either end, as the car may crash during the initial run.
        random_interp = (np.random.random_sample() * 0.4) + 0.3
        
        # Pick a random direction to face
        random_direction_interp = np.random.random_sample()

        # Compute the starting point of the car
        random_line = self.__road_points[random_line_index]
        random_start_point = list(random_line[0])
        random_start_point[0] += (random_line[1][0] - random_line[0][0])*random_interp
        random_start_point[1] += (random_line[1][1] - random_line[0][1])*random_interp

        # Compute the direction that the vehicle will face
        # Vertical line
        if (np.isclose(random_line[0][1], random_line[1][1])):
            if (random_direction_interp > 0.5):
                random_direction = (0,0,0)
            else:
                random_direction = (0, 0, math.pi)
        # Horizontal line
        elif (np.isclose(random_line[0][0], random_line[1][0])):
            if (random_direction_interp > 0.5):
                random_direction = (0,0,math.pi/2)
            else:
                random_direction = (0,0,-1.0 * math.pi/2)
        # The z coordinate is always zero
        random_start_point[2] = -1
        
        return (random_start_point, random_direction)

    # Gets an image from AirSim
    def __get_image(self):
        # Get Segmentation image
        image_response_seg = self.__car_client.simGetImages([ImageRequest(0, AirSimImageType.Infrared, False, False)])[0]
        image1dSeg = np.fromstring(image_response_seg.image_data_uint8, dtype=np.uint8)
        image_seg = image1dSeg.reshape(image_response_seg.height, image_response_seg.width, 4)

        # We actually can reduce the segmentation to mask and convert image_depth to gray-scale
        return image_seg[76:135,0:255,0:3].astype(int)

    # Appends a sample to a ring buffer.
    # If the appended example takes the size of the buffer over buffer_size, the example at the front will be removed.
    def __append_to_ring_buffer(self, item, buffer, buffer_size):
        if (len(buffer) >= buffer_size):
            buffer = buffer[1:]
        buffer.append(item)
        return buffer

    # Adds a set of examples to the replay memory
    def __add_to_replay_memory(self, field_name, data):
        if field_name not in self.__experiences:
            self.__experiences[field_name] = data
        else:
            self.__experiences[field_name] += data
            start_index = max(0, len(self.__experiences[field_name]) - self.__replay_memory_size)
            self.__experiences[field_name] = self.__experiences[field_name][start_index:]

    # Set vehicle contral signal
    def __set_vehicle_control(self, steering, throttle, brake):
    	self.__car_controls.steering = steering
        self.__car_controls.throttle = throttle
        self.__car_controls.brake = brake
        self.__car_client.setCarControls(self.__car_controls)

    # A helper function to make a directory if it does not exist
    def __make_dir_if_not_exist(self, directory):
        if not (os.path.exists(directory)):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

