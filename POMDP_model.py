import numpy as np
import random as rand
from collections import defaultdict

class pomdp_model:
    
    """
    Defines everything that goes into a POMDP
    """
    
    def __init__(self,S,A,T,R,O,Z,gamma,isterminal):
        self.S = S # states space
        self.A = A # action space
        self.T = T # transition probabilities
        self.R = R # rewards
        self.O = O # observation space
        self.Z = Z # observation probabilities
        self.gamma = gamma # decay
        self.isterminal = isterminal # check if state terminal
        
class pomcp(pomdp_model):
    
    """
    Class for solving POMDPs using POMCP montecarlo approach
    """
    
    def __init__(self,S,A,T,R,O,Z,gamma,isterminal,N,Q,d,m,c,V):  
        self.N = N # number of visits N(h,a)
        self.Q = Q # Q(h,a)
        self.d = d # depth
        self.m = m # number of simulations
        self.c = c # exploration constant
        self.V = V # Value function V(h)
        self.levels = 0 # used to keep track of how far down we are
        pomdp_model.__init__(self,S,A,T,R,O,Z,gamma,isterminal) # inherit POMDP attributes

    def bonus(self,Nha,Nh):
    
        """
        Bonus function for MCTS
        
        Inputs:
            Nha - number of times visiting history-action pair
            Nh - number of times taking any action from this point
            
        Outputs:
            outputs bonus for choosing action
        """
    
        if Nha == 0:
            return np.inf
        else:
            return np.sqrt(np.log(Nh)/Nha)

    def explore(self,h):
    
        """
        Function used to explore space
        
        Inputs:
            h - history
            
        Outputs:
            outputs argmax of all actions
        """
        
        Nh = 0.0
        for a in self.A:
            Nh += self.N[h][a]
        
        val = defaultdict(dict)
        for a in self.A:
            val[a] = self.Q[h][a]+self.c*self.bonus(self.N[h][a],Nh)
            
        return max(val,key=val.get)

    def simulate(self,s,h,d):
    
        """
        Recursive function used to simulate a rollout
        
        Inputs:
            s - state
            h - history
            d - depth to seach
            
        Outputs:
            outputs estimated value
        """
        
        # Update how far we've traversed (in depth)
        self.levels = self.d - d
        
        if d <= 0 or self.isterminal(s):
            return self.V[s]
        
        try:
            self.N[h][self.A[0]]
            haskey = True
        except:
            haskey = False
            
        if haskey == False:
             for a in self.A:
                 self.N[h][a] = 0
                 self.Q[h][a] = 0.0
             return self.V[s]
             
        a = self.explore(h)
        sp = self.T(s,a)
        r = self.R(s,a)
        o = self.Z(sp,a)
        q = r+self.gamma*self.simulate(sp,h+a+o,d-1)
        self.N[h][a] += 1
        self.Q[h][a] += (q-self.Q[h][a])/self.N[h][a]
        return q

    def pomcp_solver(self,b,h=""):
    
        """
        Online montecarlo-based POMDP solver
        
        Inputs:
            b - belief
            h - history
            
        Outputs:
            outputs best action
        """
        
        for i in range(self.m):
            s = rand.choices(self.S,b)[0]
            self.simulate(s,h,self.d)
            
        val = self.Q[h]
        return max(val,key = val.get)
        
        
        
        