from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

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
        self.max_push_effort = 5.0  # the range of force applied to the cartpole
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

        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

        # initialise envs and state tensors
        self.envs, self.num_bodies = self.create_envs()

        # current actions
        self.goal_rot = torch.ones((self.args.num_envs, 2), device=self.args.sim_device) * 1.57

        self.actions_tensor = torch.zeros((self.args.num_envs, self.num_bodies, 3), device=self.args.sim_device, dtype=torch.float)
        self.torques_tensor = torch.zeros((self.args.num_envs, self.num_bodies, 3), device=self.args.sim_device, dtype=torch.float)

        rb_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(rb_tensor)
        self.rb_pos = self.rb_states[:, 0:3].view(self.args.num_envs, self.num_bodies, 3)
        self.rb_vels = self.rb_states[:, 7:10].view(self.args.num_envs, self.num_bodies, 3)
        self.rb_rot = self.rb_states[:, 3:7].view(self.args.num_envs, self.num_bodies, 4)
        self.root_angvels = self.rb_states[:, 10:13].view(self.args.num_envs, self.num_bodies, 3)

        # generate viewer for visualisation
        if not self.args.headless:
            self.viewer = self.create_viewer()

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

        # add cartpole asset
        asset_root = 'assets'
        asset_file = 'urdf/pointer.urdf'
        asset_options = gymapi.AssetOptions()
        #asset_options.fix_base_link = True
        point_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        num_bodies = self.gym.get_asset_rigid_body_count(point_asset)

        # define cartpole pose
        pose = gymapi.Transform()
        pose.p.z = self.ball_height   # generate the cartpole 1m from the ground
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # generate environments
        self.point_handles = []
        envs = []
        print(f'Creating {self.args.num_envs} environments.')
        for i in range(self.args.num_envs):
            # create env
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # add cartpole here in each environment
            point_handle = self.gym.create_actor(env, point_asset, pose, "pointmass", i, 1, 0)
            #self.gym.set_actor_dof_properties(env, cartpole_handle, dof_props)

            envs.append(env)
            self.point_handles.append(point_handle)

        return envs, num_bodies

    def create_viewer(self):
        # create viewer for debugging (looking at the center of environment)
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(10, 0.0, 5)
        cam_target = gymapi.Vec3(-1, 0, 0)
        self.gym.viewer_camera_look_at(viewer, self.envs[self.args.num_envs // 2], cam_pos, cam_target)
        return viewer

    def get_obs(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.args.num_envs)

        self.gym.refresh_rigid_body_state_tensor(self.sim)

        roll, pitch, yaw = get_euler_xyz(self.rb_rot[env_ids, 0, :])

        pitch = torch.where(pitch > torch.pi, pitch - 2*torch.pi, pitch)
        yaw = torch.where(yaw > torch.pi, yaw - 2*torch.pi, yaw)
        roll = torch.where(roll > torch.pi, roll - 2*torch.pi, roll)

        # self.obs_buf[env_ids, 0] = self.rb_rot[env_ids, 0, 0]
        # self.obs_buf[env_ids, 1] = self.rb_rot[env_ids, 0, 1]
        # self.obs_buf[env_ids, 2] = self.rb_rot[env_ids, 0, 2]
        # self.obs_buf[env_ids, 3] = self.rb_rot[env_ids, 0, 3]

        self.obs_buf[env_ids, 0] = roll
        self.obs_buf[env_ids, 1] = pitch
        self.obs_buf[env_ids, 2] = yaw

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
        vel = self.rb_vels[env_ids, 0]
        xv, yv, zv = globalToLocalRot(roll, pitch, yaw, vel[env_ids,0], vel[env_ids,1], vel[env_ids,2])
        self.obs_buf[env_ids, 12] = xv
        self.obs_buf[env_ids, 13] = yv
        self.obs_buf[env_ids, 14] = zv

        #print(vel[0])
        #print(xv[0], yv[0], zv[0])

        # angular velocities
        ang_vel = self.root_angvels[env_ids, 0]
        xw, yw, zw = globalToLocalRot(roll, pitch, yaw, ang_vel[env_ids,0], ang_vel[env_ids,1], ang_vel[env_ids,2])
        self.obs_buf[env_ids, 15] = xw
        self.obs_buf[env_ids, 16] = yw
        self.obs_buf[env_ids, 17] = zw

        #print(xw[0], yw[0], zw[0])

        # absolute Z
        self.obs_buf[env_ids, 18] = self.rb_pos[env_ids,0,2]

    def get_reward(self):
        # retrieve environment observations from buffer
        x       = self.obs_buf[:, 9]
        y       = self.obs_buf[:, 10]
        z       = self.obs_buf[:, 11]

        z_abs   = self.obs_buf[:, 18]

        #print(x[0], y[0], z[0], '\n')

        yaw     = self.obs_buf[:, 2]
        pitch   = self.obs_buf[:, 1]
        roll    = self.obs_buf[:, 0]

        # print(roll[0], pitch[0], yaw[0])

        # qx      = self.obs_buf[:, 0]
        # qy      = self.obs_buf[:, 1]
        # qz      = self.obs_buf[:, 2]
        # qw      = self.obs_buf[:, 3]

        omegax  = self.obs_buf[:, 15]
        omegaz  = self.obs_buf[:, 16]
        omegay  = self.obs_buf[:, 17]
     
        self.reward_buf[:], self.reset_buf[:] = compute_point_reward(
            x, y, z, z_abs, roll, pitch, yaw, omegaz, omegay, omegax, self.goal_rot[:,0], self.goal_rot[:,1], self.actions_tensor[:,0,0] ,self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
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

        rot_goals = (torch.rand((len(env_ids)), device=self.args.sim_device) - 0.5) * 2 * math.pi


        self.rb_pos[env_ids, :] = positions[:]


        self.rb_rot[env_ids, 0, :] = quat_from_euler_xyz(rotations[:,0], rotations[:,1], rotations[:,2])

        velocities[..., 0:3] = 0
        self.rb_vels[env_ids, :] = velocities[..., 0:3]
        self.root_angvels[env_ids, :] = velocities[..., 3:6]

        self.goal_pos[env_ids, :] = pos_goals[:]
        
        #self.goal_rot[env_ids] = rot_goals[:]
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.rb_states[::2].contiguous()),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

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
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

    def exit(self):
        # close the simulator in a graceful way
        if not self.args.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def step(self, actions):
        actions2 = actions.to(self.args.sim_device).reshape((self.args.num_envs, self.num_act)) * self.max_push_effort
        
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

        

        self.actions_tensor[:,0,0] = 0.5 * FK_X + actions2[:,0] 
        self.actions_tensor[:,1,0] = 0.5 * FK_X + actions2[:,0] 

        self.actions_tensor[:,0,1] = 0.5 * FK_Y
        self.actions_tensor[:,1,1] = 0.5 * FK_Y

        self.actions_tensor[:,0,2] = 0.5 * FK_Z
        self.actions_tensor[:,1,2] = 0.5 * FK_Z

        self.torques_tensor[:,0,0] = TK_X + actions2[:,1] 
        self.torques_tensor[:,0,1] = TK_Y + actions2[:,2] 
        self.torques_tensor[:,0,2] = TK_Z + actions2[:,3] 

        #print(actions_tensor[0])
        forces = gymtorch.unwrap_tensor(self.actions_tensor)
        torques = gymtorch.unwrap_tensor(self.torques_tensor)
       
        self.gym.apply_rigid_body_force_tensors(self.sim, forces, torques, gymapi.LOCAL_SPACE)

        # simulate and render
        self.simulate()
        if not self.args.headless:
            self.render()

        # reset environments if required
        self.progress_buf += 1

        self.get_obs()
        self.get_reward()


# define reward function using JIT
#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_point_reward(x_pos, y_pos, z_pos, z_abs, roll, pitch, yaw, wz, wy, wx, yaw_g, pitch_g, x_action, reset_dist, reset_buf, progress_buf, max_episode_length):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    sqr_dist = (x_pos)**2 + (y_pos)**2 + (z_pos)**2

    # Proximity reward
    A1 = 0.55
    B1 = (2 + torch.log(A1))/(6**2)

    proximity_rew_gauss = (A1 / torch.exp(-B1)) * torch.exp(- B1 * sqr_dist)
    proximity_rew = torch.where(sqr_dist > 1, proximity_rew_gauss, A1)

    # Angle reward
    #angle_rew = - (torch.abs(yaw) + torch.abs(pitch) + torch.abs(roll)) * proximity_rew / (A1 * 2) 
    B4 = 0.5
    A4 = 0.3

    angle_rew = proximity_rew * torch.exp(- B4 * (pitch**2 + roll**2 + yaw**2)) 
    #angle = torch.acos(2*(0*qx + 0*qy + 0*qz + 1*qw)**2 - 1)
    #angle_rew = torch.exp(- B4 * angle)

    #print(angle_rew[0])

    # Rotation reward
    WZ_LIM = 30

    A2 = 0.15
    #B2 = torch.log(10/(A2) + 1)/WZ_LIM**2
    B2 = 0.01

    #rot_rew = - (torch.exp(B2 * wz**2) - 1) - A2 * (torch.exp(B2 * wy**2) - 1) - A2 * (torch.exp(B2 * wx**2) - 1)
    rot_rew =  (0.8 * torch.exp(- B2 * wz**2) + 0.1 * torch.exp(- B2 * wy**2) + 0.1 * torch.exp(- B2 * wx**2)) 

    # Total
    reward = proximity_rew + A4*angle_rew + A2*rot_rew

    #print(proximity_rew[0], angle_rew[0], rot_rew[0])

    #print(reward[0])
  
    # adjust reward for reset agents
    reward = torch.where(z_abs < 0.45, torch.ones_like(reward) * -200.0, reward)
    #reward = torch.where(torch.abs(x_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    #reward = torch.where(torch.abs(y_pos) > reset_dist, torch.ones_like(reward) * -200.0, reward)
    #reward = torch.where(torch.abs(wz) > 45, torch.ones_like(reward) * -200.0, reward)
    #reward = torch.where(x_action < 0, reward - 0.1, reward)
    #reward = torch.where((torch.abs(x_pos) < 0.1) & (torch.abs(y_pos) < 0.1), reward + 1, reward)

    reset = torch.where(torch.abs(sqr_dist) > 100, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(z_abs < 0.4, torch.ones_like(reset_buf), reset)
    reset = torch.where(torch.abs(wz) > 70, torch.ones_like(reset_buf), reset)
    reset = torch.where(torch.abs(wy) > 70, torch.ones_like(reset_buf), reset)
    reset = torch.where(torch.abs(wx) > 70, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script 
def get_euler(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = w * w - x * \
        x - y * y + z * z
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)def transform(roll, pitch, yaw, x, y, z):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = w * w + x * \
        x - y * y - z * z
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

@torch.jit.script 
def localToGlobalRot(roll, pitch, yaw, x, y, z):
    cosa = torch.cos(yaw)
    sina = torch.sin(yaw)

    cosb = torch.cos(pitch)
    sinb = torch.sin(pitch)

    cosy = torch.cos(roll)
    siny = torch.sin(roll)

    xp = x*cosa*cosb + y*(cosa*sinb*siny - sina*cosy) + z*(cosa*sinb*cosy + sina*siny)
    yp = x*sina*cosb + y*(sina*sinb*siny + cosa*cosy) + z*(sina*sinb*cosy - cosa*siny)
    zp = -x*sinb + y*cosb*siny + z*cosb*cosy 

    return xp, yp, zp

@torch.jit.script 
def globalToLocalRot(roll, pitch, yaw, x, y, z):
    cosa = torch.cos(yaw)
    sina = torch.sin(yaw)

    cosb = torch.cos(pitch)
    sinb = torch.sin(pitch)

    cosy = torch.cos(roll)
    siny = torch.sin(roll)

    # xp = x*cosa*cosb + y*(cosa*sinb*siny - sina*cosy) + z*(cosa*sinb*cosy + sina*siny)
    # yp = x*sina*cosb + y*(sina*sinb*siny + cosa*cosy) + z*(sina*sinb*cosy - cosa*siny)
    # zp = -x*sinb + y*cosb*siny + z*cosb*cosy 

    xp = x*cosa*cosb                    + y*sina*cosb                    - z*sinb
    yp = x*(cosa*sinb*siny - sina*cosy) + y*(sina*sinb*siny + cosa*cosy) + z*cosb*siny
    zp = x*(cosa*sinb*cosy + sina*siny) + y*(sina*sinb*cosy - cosa*siny) + z*cosb*cosy

    return xp, yp, zp

@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)