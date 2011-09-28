'''
Created on Nov 2, 2010

@author: arnaud
'''

from numpy import *
from tree import *
from time import time
from copy import copy
import cPickle

class TicTacToeTree(tree):
    """ This class constructs the tree of a TicTacToe game and solves it.
        Some nodes may be redundant for the order of moves is taken into account in the tree structure. """
        
    def __init__(self):
        print "Creating and solving TicTacToe ... ", 
        tstart = time()
        tree.__init__(self)
        self.height = 9
        self.walkAndSetAndEval(self.root) 
        print "done (in ", time()-tstart,  'seconds).\n'  
        
    def walkAndSetAndEval(self,n):
        #Build root node 
        if n.depht == 0:
            n.state = 0*array(range(1,10))
        #Evaluate leaf nodes
        v = self.nodeEval(n)
        if v == 1 or v == -1:
            n.value = v
            return v
        if n.depht == self.height:
            n.value = 0
            return 0
        #If not leaf node then add children
        n.child = [node() for k in range(9-n.depht)]
        for c in n.child:
            c.state = self.setState(n.state,n.child.index(c),n.depht)
            c.depht = n.depht + 1
        #Recurrence to build and evaluate the tree
        V = []
        val = 0 
        for c in n.child:
            V.append(self.walkAndSetAndEval(c))      
        if n.depht%2 == 0:
            val = max(V)
            n.value = val
        else:
            val = min(V)
            n.value = val
        return val    
                
    def setState(self,pval,i,depht):
        tampon = copy(pval)
        ind = nonzero(pval==0)
        tampon[ind[0][i]] = (-1)**depht
        return tampon
    
    def nodeEval(self, n):
        s = array([n.state[0:3],n.state[3:6],n.state[6:9]])
        if (diag(s) == 1).all() or (array([s[0,2],s[1,1],s[2,0]]) == 1).all():
            return 1
        if (diag(s) == -1).all() or (array([s[0,2],s[1,1],s[2,0]]) == -1).all():
            return -1
        for i in range(0,3):
            if (s[i,:] == 1).all() or (s[:,i] == 1).all():
                return 1
            if (s[i,:] == -1).all() or (s[:,i] == -1).all():
                return -1 
        return 0
            
    def finished(self,n):
        if n.child == []:
            return 1
        return 0
    
    def cPlayer(self,pos):
        player = 'player2'
        if pos.depht%2 == 0:
            player = 'player1'    
        pval = []
        for c in pos.child:
            pval.append(c.value)
        if player == 'player1':
            move = pos.child[pval.index(max(pval))]
        else:
            move = pos.child[pval.index(min(pval))]
        return move
    
    def hPlayer(self,pos):
        state = copy(pos.state)
        while 1:
            ans = input("Move??") 
            if state[ans-1] == 0:
                state[ans-1] = (-1)**(pos.depht)
                break
        pos = self.find(self.root,state)
        return pos 
 
class EnumTTTTStates: 
    """ This classe enumerates each the possible states of a Tic Tac Toe game only once -independently of the order of moves. 
        It returns the states and their value.
        The state/value can be used for supervised training."""
        
    def __init__(self, ttt):
        self.t = ttt
        self.collapse()
        
    def collapse(self):
        s = []
        v = []
        tstart = time()
        print " Enumerating unique states ... "
        for d in range(1,self.t.height+1):
            print 'rang ', d,
            L = self.t.walk(self.t.root,d)
            uniqueS,uniqueV = self.col(L)
            print '', len(uniqueS), 'new states'
            s.extend(copy(uniqueS))
            v.extend(copy(uniqueV))
        print "done in ", time()-tstart, " seconds."
        self.state = s
        self.value = v
        return s,v
    
    def col(self,L):
        uniqueS = []
        uniqueV = []
        Ls = [l.state for l in L]
        Lv = [l.value for l in L]
        for i in range(0,len(Ls)):
            dupl = 0
            for j in range(0,len(uniqueS)):
                if (uniqueS[j] == Ls[i]).all():
                    dupl = 1
                    break
            if dupl == 0:
                uniqueS.append(Ls[i])
                uniqueV.append(Lv[i])
        return uniqueS,uniqueV
    
    def saveAs(self, filename):
        basename = "../res/"
        path = basename + filename
        dico = {'states':self.state,'values':self.value}
        cPickle.dump(dico,open(filename,'wb'))

def play(t):
    c = input(""" 1. Human Fist, 2. Computer First \n Choice? """)            
    pos = t.root
    if c == 2:
        pos = t.cPlayer(pos)
        pos.show()
    while 1:
        pos = t.hPlayer(pos)
        if t.finished(pos):
            pos.show()
            break
        pos = t.cPlayer(pos)
        pos.show()
        if t.finished(pos):
            break    
    if (pos.value == 1 and c != 2) or(pos.value == -1 and c == 2):
        print 'Human wins !'
    elif (pos.value == -1 and c != 2) or(pos.value == 1 and c == 2):
        print 'Computer wins!'
    else:
        print 'Draw!'
                
if __name__ == '__main__': 
    t = TicTacToeTree()
    play(t)
    
    #Very slow
    e = EnumTTTTStates(t)
    print "Number of unique states: ", len(e.state)
    e.saveAs('unikStates')
    
            