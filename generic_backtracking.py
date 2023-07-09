# If there is a question that requires "try out everything", then it probably is backtracking

# https://leetcode.com/problems/expression-add-operators
"""
If you've looked at the question's statement and the examples that are given in the question, 
you would realize that there is an example where the digits are "105" and the target value is 5. 
For this particular example, there are two expressions given to us and they are 1*0+5 and 10-5.

The second expression is something that you need to look out for before getting to solve this question because this complicates things a bit. 
Although the number of operators are defined for us i.e. 3 different binary operators, but the number of operands are not really well defined for us.

PS: Go through the solution in detail for better understanding, https://leetcode.com/problems/expression-add-operators/solution/
"""
class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        if not num:
            return []
        
        # DFS method1
        def backtracking(idx=0, path='', value=0, prev=None):            
            if idx == len(num) and value == target:
                rtn.append(path)
                return
            
            # Try every number from idx(starts at 0) to end of string
            for i in range(idx+1, len(num) + 1):
                # Form the number first
                tmp = int(num[idx: i])
                
                if i == idx + 1 or (i > idx + 1 and num[idx] != '0'):
                    # If it is the first number evaluated, proceed to next index i
                    if prev is None :
                        backtracking(i, num[idx: i], tmp, tmp)
                    # If its not the first number, try ALL possible operators
                    else:
                        backtracking(i, path+'+'+num[idx: i], value + tmp, tmp)
                        # Negate the prev value for negative numbers, to work for * operator
                        backtracking(i, path+'-'+num[idx: i], value - tmp, -tmp)
                        backtracking(i, path+'*'+num[idx: i], value - prev + prev*tmp, prev*tmp)
        
        rtn = []
        backtracking()
        return rtn


# https://leetcode.com/problems/word-search/
# Solution: https://leetcode.com/problems/word-search/discuss/27660/Python-dfs-solution-with-comments

class Solution:
    def exist(self, board, word):

        '''return true if word is empty'''

        if not word: 
            return True
        
        # return false if board is empty
        if not board:
            return False
        
        m, n, w = len(board), len(board[0]), len(word) - 1
        # inside board, i, j represent the coordination of current cell to be checked (i meaning row, j meaning column) against the kth character of the 'word' 

        def dfs(i, j, k):
            # backtrack if it hits beyond the edges of the 'board'
            if i < 0 or i >= m or j < 0 or j >= n:
                return False
        
            # backtrack if the cell already visited
            if board[i][j] == '#':
                return False
        
            # backtrack if the cell does not match the current character of the 'word'
            if board[i][j] != word[k]:
                return False        

            # now that we have reached here, it means the cell matches the current character of the 'word'. 
            # just check if it was the last successful checking required (k==w). 
            # if so, we are done: the 'word' matches completely, so return True
            if k == w:
                return True
                    
            # if still not the last character then...
            # save the cell in case we need to backtrack later.  mark the cell as '#' (meaning visited)
            tmp = board[i][j] 
            board[i][j] = '#'

            # so far so good up to the kth character of the 'word' (meaning everything matched up to the kth character)
            # push the pointer one step forward (in order for the next character of the 'word' to be checked shortly)
            k += 1
            
            # inside for clause below: continue with up, down, left and right of the current cell: 
            # as soon as a match is found out, happily return true (meaning from that point on to the end of 'word', 
            # everything was matched up (due to the dfs), Yay...)
            for x, y in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                if dfs(i + x, j + y, k):
                    return True
            
            # we have reached here: meaning none of 4 potential paths (inside the for clause above) got matched up to the end. 
            # meaning the current cell is not a good candidate, 
            # so return it to the non-visited pool (by changing back to its original backed-up value 'tmp'). then return False
            board[i][j] = tmp
            return False
        
        # check the entire board, cell by cell as the starting point. 
        # the value '0' below means we will start off by checking if board[i][j] = word[0]. 
        # inside the dfs function, we push k (one step at a time) which represents the pointer to the current character (of the 'word') required to be checked.
        for i in range(m):
            for j in range(n):
                if dfs(i, j, 0):
                    return True

        # nothing matched successfully from any starting point in the 'board', so we are done, return False
        return False