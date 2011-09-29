'''
Created on Nov 15, 2010

@author: arnaud
'''

from mlp import MultilayerPerceptron
from numpy import exp, random, nonzero, c_, array, mean
from pylab import ion, clf, plot, draw
import cPickle
from copy import copy

class TTTPlayerTemplate:
    
    def __init__(self):
        self.move = {'playerX':1,'playerO':-1}
        self.game = []
        
    def possible(self,board,player):
        ind = [k for k in range(0,9) if board[k]==0]
        poss = []
        for i in ind:
            b = copy(board)
            b[i] += self.move[player]
            poss.append(b)
        return poss 
    
    def play(self,board,player):
        """ This will be overrriden in childern """
        return board

    def realV(self,board,player):
        endpos = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[2,5,8],[0,4,8],[2,4,6],[1,4,7]]
        for p in endpos:
            if (board[p,:] == self.move['playerX']).all():
                return 1.0
            if (board[p,:] == self.move['playerO']).all():
                return -1.0
        if (board!=0).all():
            return 0.
        return 'unknown'
    
    def randQ(self,E,q,beta):
        q = exp(beta*q)
        Q = list(q)
        for i in range(0,len(q)):
            Q[i] = sum(q[0:i+1])
        Q = Q/sum(q)
        x = random.random()
        ind = nonzero(Q>x)[0].min()
        return E[ind]
    
    def autoPlay(self):
        board = c_[0,0,0,0,0,0,0,0,0].T
        value = 'unknown'
        self.display(board)
        while value == 'unknown':
            board = self.play(board,'playerX')
            value = self.realV(board,'playerX')
            if value != 'unknown':
                self.display(board)
                break 
            board = self.play(board,'playerO')
            value = self.realV(board,'playerO')
            self.display(board)

    def display(self,b):
        m = array([b[0:3].T[0],b[3:6].T[0],b[6:9].T[0]])
        print m 

class UniformTTTPLayer(TTTPlayerTemplate):    
    def play(self,board,player):
        possible = self.possible(board,player)
        lp = len(possible)
        Q = [1./len(possible) for p in possible]
        choice = possible[self.randQ(range(0,lp),array(Q),1)]
        return choice

class LookupTTTPlayer(TTTPlayerTemplate):
    def __init__(self):
        TTTPlayerTemplate.__init__(self)
        self.lut = cPickle.load(open("../res/unikstates"))
        
    def play(self,board,player):
        possible = self.possible(board,player)
        Q = [self.lookupValue(p.T[0]) for p in possible]
        temp = [[possible[i],Q[i]] for i in range(len(Q))]
        random.shuffle(temp)
        possible = [t[0] for t in temp]
        Q = [t[1] for t in temp]
        if player == 'playerX':
            choice = possible[Q.index(max(Q))]
        elif player == 'playerO':
            choice = possible[Q.index(min(Q))]
        return choice

    def lookupValue(self, p):
        for i in range(len(self.lut['states'])):
            if (self.lut['states'][i] == p).all():
                return self.lut['values'][i]
  
class HumanTTTPlayer(TTTPlayerTemplate):
        
    def play(self,board,player):
        played = False
        while not played:
            c = input("Tile Number: ?")
            c = c-1
            if board[c] == 0:
                board[c] = self.move[player]
                played = True
        return board
          
class NeuralTTTPlayerTemplate(TTTPlayerTemplate):
    def __init__(self,nnodes):
        TTTPlayerTemplate.__init__(self)
        self.V = MultilayerPerceptron(nnodes)
        self.nnodes = str(nnodes)     
    
class GreedyNeuralTTTPlayer(NeuralTTTPlayerTemplate):
        
    def play(self,board,player):
        possible = self.possible(board,player)
        Q = [self.V.process(c)[0][0] for c in possible]
        if player == 'playerX':
            choice = possible[nonzero(Q==max(Q))[0]]
        elif player == 'playerO':
            choice = possible[nonzero(Q==min(Q))[0]]
        return choice

class SoftMaxNeuralTTTPlayer(NeuralTTTPlayerTemplate):    
    def play(self,board,player,beta=10):
        possible = self.possible(board,player)
        Q = [self.V.process(c)[0][0] for c in possible]
        lp = len(possible)
        if player == 'playerX':
            choice = possible[self.randQ(range(0,lp),array(Q),beta)]
        else:
            choice = possible[self.randQ(range(0,lp),-array(Q),beta)]
        return choice

class TTTSampler:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2  
    
    def sample_player1(self):
        board = c_[0,0,0,0,0,0,0,0,0].T
        s = []
        value = 'unknown'
        while value == 'unknown':
            board = self.player1.play(board,'playerX')
            s.append(copy(board))
            value =  self.player1.realV(board,'playerX')
            if value != 'unknown':
                break
            board = self.player2.play(board,'playerO')
            value = self.player2.realV(board,'playerO')
        return [s,value]
    
    def sample_player2(self):
        board = c_[0,0,0,0,0,0,0,0,0].T
        s = []
        value = 'unknown'
        while value == 'unknown':
            board = self.player1.play(board,'playerX')
            value =  self.player1.realV(board,'playerX')
            if value != 'unknown':
                break
            board = self.player2.play(board,'playerO')
            s.append(copy(board))
            value = self.player2.realV(board,'playerO')
        return [s,value]
    
    def sample_all(self):
        board = c_[0,0,0,0,0,0,0,0,0].T
        s = []
        value = 'unknown'
        while value == 'unknown':
            board = self.player1.play(board,'playerX')
            s.append(copy(board))
            value =  self.player1.realV(board,'playerX')
            if value != 'unknown':
                break
            board = self.player2.play(board,'playerO')
            s.append(copy(board))
            value = self.player2.realV(board,'playerO')
        return [s,value]

class ReinforcementLearning:
    def __init__(self, player1, player2): 
        self.player1 = player1
        self.player2 = player2
        self.sampler = TTTSampler(player1, player2)
        self.anim = Animate()
           
    def MC_RL_player2(self,n):
        for k in range(0,n):
            if n%(n/100)==0:
                print 'RL:', 100*k/n, '%'
            [s,value] = self.sampler.sample_player2()
            for state in s:
                self.player2.V.learn(state,value)
                print state.T, value, self.player2.V.process(state).T
            self.anim.anim(value)
                    
    def TD_RL_player2(self, n):                
        for k in range(0,n):
            if n%(n/100)==0:
                print 'RL:', 100*k/n, '%'
            [s,value] = self.sampler.sample_player2()
            for i in range(0,len(s)-1):
                o = self.player2.V.process(s[i])
                n = self.player2.V.process(s[i+1])
                value = o + 0.05*(n-o)
                self.V.learn(s[i],value)
                print s[i].T, value, self.player2.V.process(s[i]).T
            self.anim.anim(value)
                
    def MC_RL_player1(self,n):
        for k in range(0,n):
            if n%(n/100)==0:
                print 'RL:', 100*k/n, '%'
            [s,value] = self.sampler.sample_player1()
            for state in s:
                self.player1.V.learn(state,value)
                print state.T, value, self.player1.V.process(state).T
            self.anim.anim(value)
                    
    def TD_RL_player1(self, n):                
        for k in range(0,n):
            if n%(n/100)==0:
                print 'RL:', 100*k/n, '%'
            [s,value] = self.sampler.sample_player1()
            for i in range(0,len(s)-1):
                o = self.player1.V.process(s[i])
                n = self.player1.V.process(s[i+1])
                value = o + 0.05*(n-o)
                self.V.learn(s[i],value)
                print s[i].T, value, self.player1.V.process(s[i]).T
            self.anim.anim(value)

class Animate: 
    def __init__(self):
        self.game = []     
        self.moyg = []
        ion()
        
    def anim(self,value):
        self.game.append(value)
        clf()
        if len(self.game)%500 == 0:
            y = mean(self.game)
            self.game = []
            self.moyg.append(y)
            x = range(0,500*len(self.moyg),500)            # x-array
            plot(x,self.moyg)
            draw()
#        if len(self.game)%100000 == 0:
#            savefig('RLcurve'+type+self.nnodes)

class PlayTTT:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
      
    def play(self):
        winner = self._play()
        print "Winner: ", winner 
        
    def _play(self):
        board = c_[0,0,0,0,0,0,0,0,0].T
        value = 'unknown'
        while value == 'unknown':
            board = self.player1.play(board,'playerX')
            value =  self.player1.realV(board,'playerX')
            self.show(board)
            if value != 'unknown':
                break
            board = self.player2.play(board,'playerO')
            value = self.player2.realV(board,'playerO')
            self.show(board)
        return value
        
    def show(self,b):
        m = array([b[0:3].T[0],b[3:6].T[0],b[6:9].T[0]])
        print m 

if __name__ == "__main__":
    
#    game = PlayTTT(LookupTTTPlayer(), HumanTTTPlayer())
#    game.play()
    
    p1 = LookupTTTPlayer()
    p2 = SoftMaxNeuralTTTPlayer([9,40,1])
    
    rl = ReinforcementLearning(p1,p2)
    rl.MC_RL_player2(100000)
    

    


                
