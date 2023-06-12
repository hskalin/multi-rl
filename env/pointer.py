from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

from utils.torch_jit_utils import *
import sys

import torch
import math

class Pointer:
    def __init__(self, args):
        self.args = args

        # configure sim (gravity is pointing down)
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = True

        # set simulation parameters (we use PhysX engine by default, these parameters are from the example file)
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.rest_offset = 0.001
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.use_gpu = True

        # task-specific parameters
        self.num_obs = 19  # pole_angle + pole_vel + cart_vel + cart_pos
        self.num_act = 4  # force applied on the pole (-1 to 1)
        self.reset_dist = 10.0  # when to reset
        self.max_push_effort = 5.0  # the range of force applied to the pointer
        self.max_episode_length = 1000  # maximum episode length

        self.ball_height = 4
        self.goal_lim = 4

        # allocate buffers
        self.obs_buf = torch.zeros((self.args.num_envs, self.num_obs), device=self.args.sim_device)
        self.reward_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device)
        self.reset_buf = torch.ones(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.args.num_envs, device=self.args.sim_device, dtype=torch.long)

        self.goal_pos = (torch.rand((self.args.num_envs, 3), device=self.args.sim_device) - 0.5) * 2*self.goal_lim
        self.goal_pos[..., 2] += self.goal_lim + 1

        self.goal_rot = torch.zeros((self.args.num_envs, 3), device=self.args.sim_device)

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        # initialise envs and state tensors
        self.envs, self.num_bodies = self.create_envs()

        rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(rb_tensor)
        self.rb_pos = self.rb_states[:, 0:3].view(self.args.num_envs, self.num_bodies, 3)
        self.rb_rot = self.rb_states[:, 3:7].view(self.args.num_envs, self.num_bodies, 4)
        self.rb_lvels = self.rb_states[:, 7:10].view(self.args.num_envs, self.num_bodies, 3) 
        self.rb_avels = self.rb_states[:, 10:13].view(self.args.num_envs, self.num_bodies, 3)

        # storing tensors for visualisations
        self.actions_tensor = torch.zeros((self.args.num_envs, self.num_bodies, 3), device=self.args.sim_device, dtype=torch.float)
        self.torques_tensor = torch.zeros((self.args.num_envs, self.num_bodies, 3), device=self.args.sim_device, dtype=torch.float)

        # generate viewer for visualisation
        # if not self.args.headless:
        #     self.viewer = self.create_viewer()
        #     self.gym.subscribe_viewer_keyboard_event(
        #         self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        #     self.gym.subscribe_viewer_keyboard_event(
        #         self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

        # self.enable_viewer_sync = False
        self.set_viewer()

        # step simulation to initialise tensor buffers
        self.gym.prepare_sim(self.sim)
        self.reset()

    def create_envs(self):
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.restitution = 1
        plane_params.static_friction = 0
        plane_params.dynamic_friction = 0
        self.gym.add_ground(self.sim, plane_params)

        # define environment space (for visualisation)
        spacing = 4.0
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)
        num_per_row = int(np.sqrt(self.args.num_envs))

        # add pointer asset
        asset_root = 'assets'
        asset_file = 'urdf/pointer.urdf'
        asset_options = gymapi.AssetOptions()
        point_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        num_bodies = self.gym.get_asset_rigid_body_count(point_asset)

        # define pointer pose
        pose = gymapi.Transform()
        pose.p.z = self.ball_height   # generate the pointer 1m from the ground
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # generate environments
        self.pointer_handles = []
        envs = []
        print(f'Creating {self.args.num_envs} environments.')
        for i in range(self.args.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add pointer here in each environment
            point_handle = self.gym.create_actor(env, point_asset, pose, "pointmass", i, 1, 0)

            envs.append(env)
            self.pointer_handles.append(point_handle)

        return envs, num_bodies

    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(10, 0.0, 5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)
        return viewer
    
    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.args.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 15.0, 0.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.args.num_envs)

        # refreshes the rb state tensor with new values
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        roll, pitch, yaw = get_euler_xyz(self.rb_rot[env_ids, 0, :])

        # maps rpy from -pi to pi
        pitch = torch.where(pitch > torch.pi, pitch - 2*torch.pi, pitch)
        yaw = torch.where(yaw > torch.pi, yaw - 2*torch.pi, yaw)
        roll = torch.where(roll > torch.pi, roll - 2*torch.pi, roll)

        # angles
        self.obs_buf[env_ids, 0] = roll     - self.goal_rot[env_ids, 0]
        self.obs_buf[env_ids, 1] = pitch    - self.goal_rot[env_ids, 1]
        self.obs_buf[env_ids, 2] = yaw      - self.goal_rot[env_ids, 2]

        # rotations
        sine_x = torch.sin(roll)
        cosine_x = torch.cos(roll)

        sine_y = torch.sin(pitch)
        cosine_y = torch.cos(pitch)

        sine_z = torch.sin(yaw)
        cosine_z = torch.cos(yaw)

        self.obs_buf[env_ids, 3] = sine_x[env_ids]
        self.obs_buf[env_ids, 4] = cosine_x[env_ids]

        self.obs_buf[env_ids, 5] = sine_y[env_ids]
        self.obs_buf[env_ids, 6] = cosine_y[env_ids]

        self.obs_buf[env_ids, 7] = sine_z[env_ids]
        self.obs_buf[env_ids, 8] = cosine_z[env_ids]

        # relative xyz pos
        pos = self.goal_pos[env_ids] - self.rb_pos[env_ids, 0]

        xp, yp, zp = globalToLocalRot(roll, pitch, yaw, pos[env_ids,0], pos[env_ids,1], pos[env_ids,2])
        self.obs_buf[env_ids, 9] = xp
        self.obs_buf[env_ids, 10] = yp
        self.obs_buf[env_ids, 11] = zp
       
        # relative xyz vel
        vel = self.rb_lvels[env_ids, 0]

        xv, yv, zv = globalToLocalRot(roll, pitch, yaw, vel[env_ids,0], vel[env_ids,1], vel[env_ids,2])
        self.obs_buf[env_ids, 12] = xv
        self.obs_buf[env_ids, 13] = yv
        self.obs_buf[env_ids, 14] = zv

        # angular velocities
        ang_vel = self.rb_avels[env_ids, 0]

        xw, yw, zw = globalToLocalRot(roll, pitch, yaw, ang_vel[env_ids,0], ang_vel[env_ids,1], ang_vel[env_ids,2])
        self.obs_buf[env_ids, 15] = xw
        self.obs_buf[env_ids, 16] = yw
        self.obs_buf[env_ids, 17] = zw

        # absolute Z
        self.obs_buf[env_ids, 18] = self.rb_pos[env_ids,0,2]

    def get_reward(self):
        # retrieve environment observations from buffer
        x       = self.obs_buf[:, 9]
        y       = self.obs_buf[:, 10]
        z       = self.obs_buf[:, 11]

        z_abs   = self.obs_buf[:, 18]

        yaw     = self.obs_buf[:, 2]
        pitch   = self.obs_buf[:, 1]
        roll    = self.obs_buf[:, 0]

        omegax  = self.obs_buf[:, 15]
        omegaz  = self.obs_buf[:, 16]
        omegay  = self.obs_buf[:, 17]
     
        self.reward_buf[:], self.reset_buf[:] = compute_point_reward(
                                                x, y, z, 
                                                z_abs, 
                                                roll, pitch, yaw, 
                                                omegax, omegay, omegaz,
                                                self.reset_dist, 
                                                self.reset_buf, 
                                                self.progress_buf, 
                                                self.max_episode_length
        )

    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) == 0:
            return

        # randomise initial positions and velocities
        positions = torch.zeros((len(env_ids), self.num_bodies, 3), device=self.args.sim_device)
        positions[:,:,2] = self.ball_height

        velocities = 2 * (torch.rand((len(env_ids), self.num_bodies, 6), device=self.args.sim_device) - 0.5)
        
        rotations = (torch.rand((len(env_ids), 3), device=self.args.sim_device) - 0.5) * 2 * math.pi

        pos_goals = (torch.rand((len(env_ids), 3), device=self.args.sim_device) - 0.5) * 2*self.goal_lim
        pos_goals[:,2] += self.goal_lim + 1

        # set random pos, rot, vels
        self.rb_pos[env_ids, :] = positions[:]

        self.rb_rot[env_ids, 0, :] = quat_from_euler_xyz(rotations[:,0], rotations[:,1], rotations[:,2])

        self.rb_lvels[env_ids, :] = velocities[..., 0:3]
        self.rb_avels[env_ids, :] = velocities[..., 3:6]

        self.goal_pos[env_ids, :] = pos_goals[:]
        
        # selectively reset the environments
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.rb_states[::2].contiguous()),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # clear relevant buffers
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # refresh new observation after reset
        self.get_obs()

    def simulate(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def render(self):
        # update viewer
        # self.gym.step_graphics(self.sim)
        # self.gym.draw_viewer(self.viewer, self.sim, True)
        # self.gym.sync_frame_time(self.sim)
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

        # step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)

            num_lines = 3
            line_color = [[0,0,0], [0,0,0], [200,0,0]]

            roll, pitch, yaw = get_euler_xyz(self.rb_rot[:, 0, :])
            xa, ya, za = localToGlobalRot(roll, pitch, yaw, self.actions_tensor[:,0,0], self.actions_tensor[:,0,1], self.actions_tensor[:,0,2])

            for i, envi in enumerate(self.envs):

                vertices = [[self.goal_pos[i,0].item(), self.goal_pos[i,1].item(),0], 
                            [self.goal_pos[i,0].item(), self.goal_pos[i,1].item(), self.goal_pos[i,2].item()],
                            [self.goal_pos[i,0].item(), self.goal_pos[i,1].item(), self.goal_pos[i,2].item()], 
                            [self.goal_pos[i,0].item()+ math.cos(self.goal_rot[i,0].item()), self.goal_pos[i,1].item()+math.sin(self.goal_rot[i,0].item()), self.goal_pos[i,2].item()],
                            [self.rb_pos[i,0,0].item(), self.rb_pos[i,0,1].item(), self.rb_pos[i,0,2].item()],
                            [self.rb_pos[i,0,0].item()-xa[i].item(), self.rb_pos[i,0,1].item()-ya[i].item(), self.rb_pos[i,0,2].item()-za[i].item()]]
                
                self.gym.add_lines(self.viewer, envi, num_lines, vertices, line_color)


            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.clear_lines(self.viewer)

            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

        else:
                self.gym.poll_viewer_events(self.viewer)

    def exit(self):
        # close the simulator in a graceful way
        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def step(self, actions):
        actions = actions.to(self.args.sim_device).reshape((self.args.num_envs, self.num_act)) * self.max_push_effort
        
        # viscosity multiplier
        Q = 3

        # friction = kv^2
        # Ks for forces :
        FK_X = - Q * 0.0189 * torch.sign(self.obs_buf[:, 12]) * self.obs_buf[:, 12]**2
        FK_Y = - Q * 0.0472 * torch.sign(self.obs_buf[:, 13]) * self.obs_buf[:, 13]**2
        FK_Z = - Q * 0.0943 * torch.sign(self.obs_buf[:, 14]) * self.obs_buf[:, 14]**2

        # Ks for torques
        TK_X = - Q * 0.0004025 * torch.sign(self.obs_buf[:, 15]) * self.obs_buf[:, 15]**2
        TK_Y = - Q * 0.003354 * torch.sign(self.obs_buf[:, 16]) * self.obs_buf[:, 16]**2
        TK_Z = - Q * 0.003354 * torch.sign(self.obs_buf[:, 17]) * self.obs_buf[:, 17]**2

        self.actions_tensor[:,0,0] = 0.5 * FK_X + actions[:,0] 
        self.actions_tensor[:,1,0] = 0.5 * FK_X + actions[:,0] 

        self.actions_tensor[:,0,1] = 0.5 * FK_Y
        self.actions_tensor[:,1,1] = 0.5 * FK_Y

        self.actions_tensor[:,0,2] = 0.5 * FK_Z
        self.actions_tensor[:,1,2] = 0.5 * FK_Z

        self.torques_tensor[:,0,0] = TK_X + actions[:,1] 
        self.torques_tensor[:,0,1] = TK_Y + actions[:,2] 
        self.torques_tensor[:,0,2] = TK_Z + actions[:,3] 

        # unwrap tensors
        forces = gymtorch.unwrap_tensor(self.actions_tensor)
        torques = gymtorch.unwrap_tensor(self.torques_tensor)
       
        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, forces, torques, gymapi.LOCAL_SPACE)

        # simulate and render
        self.simulate()
        if not self.args.headless:
            self.render()

        # reset environments if required
        self.progress_buf += 1

        self.get_obs()
        self.get_reward()


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_point_reward(x_pos, y_pos, z_pos, z_abs, roll, pitch, yaw, wx, wy, wz, reset_dist, reset_buf, progress_buf, max_episode_length):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # square distance from goal
    sqr_dist = (x_pos)**2 + (y_pos)**2 + (z_pos)**2

    # Proximity reward
    A1 = 0.55
    B1 = (2 + torch.log(A1))/(6**2)

    proximity_rew_gauss = (1 / torch.exp(-B1)) * torch.exp(- B1 * sqr_dist)
    proximity_rew = torch.where(sqr_dist > 1, proximity_rew_gauss, 1)

    # Angle reward
    A2 = 0.3
    B2 = 0.5

    angle_rew = proximity_rew * torch.exp(- B2 * (pitch**2 + roll**2 + yaw**2)) 

    # Rotation reward
    A3 = 0.15
    B3 = 0.01

    rot_rew =  (0.8 * torch.exp(- B3 * wz**2) + 0.1 * torch.exp(- B3 * wy**2) + 0.1 * torch.exp(- B3 * wx**2)) 

    # Total
    reward = A1*proximity_rew + A2*angle_rew + A3*rot_rew

    #print(proximity_rew[0], angle_rew[0], rot_rew[0])
    #print(reward[0])
  
    # adjust reward for reset agents
    reward = torch.where(z_abs < 0.75, torch.ones_like(reward) * -200.0, reward)
    #reward = torch.where(torch.abs(x_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    #reward = torch.where(torch.abs(y_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    #reward = torch.where(torch.abs(wz) > 45, torch.ones_like(reward) * -200.0, reward)
    #reward = torch.where(x_action < 0, reward - 0.1, reward)
    #reward = torch.where((torch.abs(x_pos) < 0.1) & (torch.abs(y_pos) < 0.1), reward + 1, reward)

    reset = torch.where(torch.abs(sqr_dist) > 100, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(z_abs < 0.8, torch.ones_like(reset_buf), reset)
    reset = torch.where(torch.abs(wz) > 70, torch.ones_like(reset_buf), reset)
    reset = torch.where(torch.abs(wy) > 70, torch.ones_like(reset_buf), reset)
    reset = torch.where(torch.abs(wx) > 70, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset