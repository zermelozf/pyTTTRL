'''
Created on Nov 2, 2010

@author: arnaud
'''

from numpy import array, nonzero
    
class node:
    def __init__(self):
        self.parent = []
        self.child = []
        self.depht = 0
        self.state = 0
        self.value = 0
        
    def addChild(self,nodes):
        self.child.extend(nodes)
        for n in nodes:
            n.parent.append(self)
            n.depht = self.depht+1
    
    def show(self):
        s = self.state
        print array([s[0:3],s[3:6],s[6:9]])
            
        
class tree(node):
    def __init__(self):
        self.root = node()
    
    def walk(self,node,d):
        if node.depht == d or node.child == []:
            return node 
        ret = []
        for c in node.child:
            r = self.walk(c,d)
            if type(r) is list:
                ret.extend(r)
            else:
                ret.append(r)
        return ret
    
    # Find recursively    
    def find(self,snode,s):
        if (snode.state == s).all():
            return snode 
        f = []
        for c in snode.child:
            f = self.find(c,s)
            if f != []:
                return f
        return []        
    
    # Find iteratively
    def itfind(self,state):
        found = self.root
        while found.child != []:
            for s in found.child:
                ind = nonzero(state==(-1)**(s.depht+1))[0][(s.depht-1)/2]
                if s.state[ind] == state[ind]:
                    found = s
                    break
            if (found.state == state).all():
                return found
        print 'No match found (intfind) for ', state
        
        
 
        