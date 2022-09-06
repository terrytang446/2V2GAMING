import torch
import numpy as np
from .planner2 import Planner, save_model, load_model
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import copy

class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None
        self.n = 0

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = load_model().to(self.device)
        self.model.eval()

        self.current_team = 0
        self.back = {0:False,1:False}
        self.back_count = {0:0,1:0}
        self.back_steer = {0:1,1:1}
        self.turn_count = {0:0,1:0}
        self.prev_loc = {0:np.int32([0,0]),1:np.int32([0,0])}
        self.prev_steer = {0:0,1:0}

    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.n = 0
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        image1 = TF.to_tensor(player_image[0]).to(self.device)
        image2 = TF.to_tensor(player_image[1]).to(self.device)
        #print(image1.shape)
        ball_locs = []
        ball_locs.append(self.model(image1).squeeze(0).detach().cpu().numpy())
        ball_locs.append(self.model(image2).squeeze(0).detach().cpu().numpy())
        
        actionList = []

        for i in range(self.num_players):
          player_front = player_state[i]['kart']['front']
          player_location = player_state[i]['kart']['location']
          #print(player_location)
          player_velocity = player_state[i]['kart']['velocity']
          front2d = [player_front[0]-player_location[0],player_front[2]-player_location[2]]

          toBall2d = [ball_locs[i][0]-200,-ball_locs[i][1]+150]

          velocity2d = [player_velocity[0],player_velocity[2]]
          velocity2dNorm = np.linalg.norm(velocity2d)

          front2dUnit = front2d  / np.linalg.norm(front2d)
          toBall2dUnit = [toBall2d[0]/200,toBall2d[1]/150]
          #toBall2dUnit = toBall2d  / np.linalg.norm(toBall2d)
          toBall2dUnit = np.clip(toBall2dUnit,-1,1)
          angleToBall = np.arcsin(toBall2dUnit[0])*180/(np.pi)
          angleToBall = angleToBall*(-1)
          #print(toBall2dUnit,angleToBall)

          # player_front = player_state[i]['kart']['front']
          # player_location = player_state[i]['kart']['location']
          # player_velocity = player_state[i]['kart']['velocity']
          # front2d = [player_front[0]-player_location[0],player_front[2]-player_location[2]]
          # toBall2d = [soccer_location[0]-player_location[0],soccer_location[2]-player_location[2]]
          # velocity2d = [player_velocity[0],player_velocity[2]]
          # velocity2dNorm = np.linalg.norm(velocity2d)
          # front2dUnit = front2d  / np.linalg.norm(front2d)
          # toBall2dUnit = toBall2d  / np.linalg.norm(toBall2d)
          # isBallLeft = np.cross(front2dUnit, toBall2dUnit) > 0
          # angleToBall = np.arccos(np.dot(front2dUnit,toBall2dUnit))*180/(np.pi)
          # if isBallLeft == False:
          #   angleToBall = angleToBall*(-1)

          if self.current_team == 0:
            if player_location[2]<0:
                self.current_team = 1
            else:
                self.current_team = 2
          #print(self.current_team)
          isToGoal = True
          if self.current_team == 1:
            isToGoal = True if front2dUnit[1]>0 else False
          else:
            isToGoal = True if front2dUnit[1]<0 else False
          
          brake_o = False
          acceleration_o = 1
          steer_o = 0
          drift_o = False
          nitro_o = True

          # tune params
          steer_threshold = 5
          drift_threshold = 50
          acceleration_threshold = 30
          velocity_threshold = 20

          if(isToGoal==False):
            if angleToBall < 0:
              angleToBall -= steer_threshold
            else:
              angleToBall += steer_threshold
          if (angleToBall > steer_threshold) :
            steer_o = -1
          elif (angleToBall < -steer_threshold):
            steer_o = 1
          else:
            steer_o = 0
          if (angleToBall > drift_threshold) or (angleToBall < -drift_threshold) :
            drift_o = True
          else:
            drift_o = False
          if (angleToBall > acceleration_threshold) or (angleToBall < -acceleration_threshold) :
            acceleration_o = 0.5
          if (velocity2dNorm > velocity_threshold):
            acceleration_o = 0.2
          
          now_loc = [player_location[0],player_location[2]]
          if self.back[i] == True:
            brake_o = True
            steer_o = self.back_steer[i]
            acceleration_o = 0
            self.back_count[i] -= 2
            if self.back_count[i] < 1 or (velocity2dNorm < 5 and (-7<now_loc[0]<1 and -57<now_loc[1]<57)):
                self.back[i] = False
          else:
            if self.prev_loc[i][0] == np.int32(now_loc)[0] and self.prev_loc[i][1] == np.int32(now_loc)[1]:
              self.back_count[i] += 5
            else:
              self.back_count[i] = 0

            if self.back_count[i] == 0:
                if angleToBall<0:
                    self.back_steer[i] = -1
                else:
                    self.back_steer[i] = 1
            if self.back_count[i] > 30:
                if velocity2dNorm > 10:
                    self.back_count[i] = 30
                    self.back_steer[i] = 0
                else:
                    self.back_count[i] = 20
                self.back[i] = True

            if (steer_o == self.prev_steer[i]):
              self.turn_count[i] += 1
            else:
              self.turn_count[i] = 0
            if self.turn_count[i] > 29 and isToGoal and (front2dUnit[0]*steer_o<0 if self.current_team == 1 else front2dUnit[0]*steer_o>0):
              #print(self.current_team,isToGoal,front2dUnit,steer_o)
              self.back_count[i] = 32
              self.turn_count[i] = 0
              self.back[i] = True
          
          self.prev_loc[i] = np.int32(now_loc)
          self.prev_steer[i] = steer_o

          #print(angleToBall)
          # if(self.n < 20):
          #   acceleration_o = 1
          #   steer_o = 0

          actionList.append(dict(acceleration=acceleration_o, steer=steer_o, brake = brake_o, drift = drift_o, nitro = nitro_o))

        # TODO: Change me. I'm just cruising straight
        self.n=self.n+1
        if(self.n==1):
          print(1)
        #print(self.n)
        return actionList
