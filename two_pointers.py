# https://leetcode.com/problems/flatten-2d-vector
"""
Design and implement an iterator to flatten a 2d vector. It should support the following operations: next and hasNext.

Vector2D iterator = new Vector2D([[1,2],[3],[4]]);

iterator.next(); // return 1
iterator.next(); // return 2
iterator.next(); // return 3
iterator.hasNext(); // return true
iterator.hasNext(); // return true
iterator.next(); // return 4
iterator.hasNext(); // return false
"""
class Vector2D:
    def __init__(self, vec2d):
        self.col = 0
        self.row = 0
        self.vec = vec2d
        
    def next(self):
        if self.hasNext():
            result = self.vec[self.row][self.col]
            self.col += 1
            return result

    def hasNext(self):
        while self.row < len(self.vec):
            if self.col < len(self.vec[self.row]):
                return True
            
            self.col = 0
            self.row += 1
            
        return False
