import numpy as np
from spinup.MultiAgentEnvMod.multiagent.core import World, Agent, Landmark
from spinup.MultiAgentEnvMod.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_landmarks = 2
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.ID = i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05      
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.ID = i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            # agent.color = np.array([0.35, 0.35, 0.85])
            col = (agent.ID) % 2
            agent.color = np.array([col*0.5, 0.5, (1-col)*0.5])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            # landmark.color = np.array([0.5, 0.5, 0.55])
            col = (landmark.ID) % 2
            landmark.color = np.array([col*0.5, 0.5, (1-col)*0.5])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-3, +3, world.dim_p)
            # agent.state.p_pos = np.array([-0.5, 0.5-agent.ID/2])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-3, +3, world.dim_p)
            # landmark.state.p_pos = np.array([0.5, 0.5-landmark.ID/2])
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agent is rewarded by distance to goal. Max reward is 0
        rew = 0
        dists = np.sqrt(np.sum(np.square(world.agents[0].state.p_pos - world.landmarks[0].state.p_pos)))
        # rew -= dists
        if dists < 1:
            rew += 1/dists
        else:
            rew -= dists


        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            if agent.ID == entity.ID: 
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)    

        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
        return np.concatenate([agent.state.p_pos] + entity_pos + other_pos)
