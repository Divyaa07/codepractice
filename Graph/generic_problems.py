import collections
from operator import contains
from collections import deque



# https://leetcode.com/problems/find-if-path-exists-in-graph/

class Solution:

    def validPath(self, n: int, edges: List[List[int]], start: int, end: int) -> bool:
        neighbors = defaultdict(list)
        for n1, n2 in edges:
            neighbors[n1].append(n2)
            neighbors[n2].append(n1)
            
        q = deque([start])
        seen = set([start])
        while q:
            node = q.popleft()            
            if node == end:
                return True            
            for n in neighbors[node]:
                if n not in seen:
                    seen.add(n)
                    q.append(n)
        return False


    def validPath(self, n: int, edges: List[List[int]], start: int, end: int) -> bool:
        neighbors = defaultdict(list)
        for n1, n2 in edges:
            neighbors[n1].append(n2)
            neighbors[n2].append(n1)
            
        def dfs(node, end, seen):
            if node == end:
                return True
            if node in seen:
                return False
            
            seen.add(node)
            for n in neighbors[node]:
                if dfs(n, end, seen):
                    return True
                
            return False

        seen = set()
        return dfs(start, end, seen)


def first_missing_positive(nums):
    n = len(nums)
    for i in range(n):
        while 1 <= nums[i] <= n and nums[i] != nums[nums[i]-1]:
            tmp = nums[nums[i]-1]
            nums[nums[i]-1] = nums[i]
            nums[i] = tmp
    for i in range(n):
        if nums[i] != i+1:
            return i+1
    return n+1



# Color graph of degree D with alternate colors
def color_graph(graph, colors):
    if node in node.neighbors:
        raise Exception('Legal coloring impossible for node with loop: %s' %
                        node.label)
    for node in graph:
        # Get the node's neighbors' colors, as a set so we
        # can check if a color is illegal in constant time
        illegal_colors = set([
            neighbor.color
            for neighbor in node.neighbors
            if neighbor.color
        ])

        # Assign the first legal color
        for color in colors:
            if color not in illegal_colors:
                node.color = color
                break


# Given a matrix, find out if it can reach from a (0,0) to end point, avoiding points
# that has 0 in it, only traverse through 1. Return min distance from source to an obstacle (value 9)
def min_distance_matrix_BFS(matrix):
    if not matrix:
        return -1
    rows = len(matrix)
    cols = len(matrix[0])
    points_to_traverse = [(0, 0, 0)]
    visited_points = [[False for c in range(cols)] for r in range(rows)]

    def get_valid_next_moves(pos):
        next_pts = []
        possible_moves = [(1,0), (-1,0), (0,1), (0, -1)]
        for move in possible_moves:
            potential_point = (pos[0]+move[0], pos[1]+move[1])
            if potential_point[0] >= 0 and potential_point[0] < rows \
                    and potential_point[1] >=0 and potential_point[1] < cols \
                    and matrix[potential_point[0]][potential_point[1]] != 0\
                    and not visited_points[potential_point[0]][potential_point[1]]:
                next_pts.append(potential_point[0], potential_point[1], pos[2]+1)
                visited_points[potential_point[0]][potential_point[1]] = True

            return next_pts

    while (points_to_traverse):
        cur_pos = points_to_traverse.pop(0)
        if matrix[cur_pos[0]][cur_pos[1]] == 9:
            return cur_pos[2]
        else:
            next_points = get_valid_next_moves(cur_pos)
            for point in next_points:
                points_to_traverse.append(point)
    return -1

def min_distance_matrix_DFS(matrix):
    # Incomplete
    if not matrix:
        return -1
    rows = len(matrix)
    cols = len(matrix[0])
    visited_points = [[False for _ in range(cols)] for _ in range(rows)]

    def helper_min_distance(row, col, min_distance):
        if not (row >= 0 and row < rows and col >=0 and col < cols):
            return 'inf'

        if matrix[row][col] == 9:
            return min_distance

        visited_points[row][col] = True
        return min(helper_min_distance(row-1, col, min_distance+1), helper_min_distance(row, col-1, min_distance+1),
               helper_min_distance(row, col+1, min_distance+1), helper_min_distance(row+1, col, min_distance+1))

    return helper_min_distance(0, 0, 0)


# https://leetcode.com/problems/number-of-islands/
# DFS Solution
def part_of_island(self, i, j, grid):
    if i < 0 or j < 0 or i == len(grid) or j == len(grid[0]) or grid[i][j] != '1':
        return
    else:
        grid[i][j] = '0'
    self.part_of_island(i, j + 1, grid)
    self.part_of_island(i, j - 1, grid)
    self.part_of_island(i + 1, j, grid)
    self.part_of_island(i - 1, j, grid)

def numIslands(self, grid):
    # Bad solution
    islands = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                islands += 1
                self.part_of_island(i,j,grid)
    return islands


# https://leetcode.com/problems/path-with-maximum-gold/
# DFS
def getMaximumGold(self, grid):
    def dfs(i, j, sum, seen):
        if i < 0 or i >= m or j < 0 or j >= n or not grid[i][j] or (i, j) in seen:
            return sum
        seen.add((i, j))
        sum += grid[i][j]
        mx = 0
        for x, y in ((i, j + 1), (i , j - 1), (i + 1, j), (i - 1, j)):
            mx = max(dfs(x, y, sum, seen), mx)
        seen.discard((i, j))
        return mx

    m, n = len(grid), len(grid[0])
    ans = 0
    for i in range(m):
        for j in range(n):
            ans = max(ans, dfs(i,j,0, set()))
    return ans


# Backtracking (Same as DFS)
class Solution:
    def getMaximumGold(grid):
        # Given a row and a column, what are all the neighbours?
        def options(row, col):
            if row > 0:
                yield (row - 1, col)
            if col > 0:
                yield (row, col - 1)
            if row < len(grid) - 1:
                yield (row + 1, col)
            if col < len(grid[0]) - 1:
                yield (row, col + 1)

        # Keep track of current gold we have, and best we've seen.
        current_gold = 0
        maximum_gold = 0

        def backtrack(row, col):
            # If there is no gold in this cell, we're not allowed to continue.
            if grid[row][col] == 0:
                return
            # Keep track of this so we can put it back when we backtrack.
            gold_at_square = grid[row][col]
            # Add the gold to the current amount we're holding.
            current_gold += gold_at_square
            # Check if we currently have the max we've seen.
            maximum_gold = max(self.maximum_gold, self.current_gold)
            # Take the gold out of the current square.
            grid[row][col] = 0
            # Consider all possible ways we could go from here.
            for neigh_row, neigh_col in options(row, col):
                # Recursively call backtrack to explore this way.
                backtrack(neigh_row, neigh_col)
            # Once we're done on this path, backtrack by putting gold back.
            current_gold -= gold_at_square
            grid[row][col] = gold_at_square

        # Start the search from each valid starting location.
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                backtrack(row, col)
        # Return the maximum we saw.
        return maximum_gold


# https://leetcode.com/problems/search-a-2d-matrix-ii
class Solution:
    def searchMatrix(self, matrix, target):
    # Method 1: Optimal (Binary search in row/col by moving across diagonal)
        # an empty matrix obviously does not contain `target` (make this check
        # because we want to cache `width` for efficiency's sake)
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False
        # cache these, as they won't change.
        height = len(matrix)
        width = len(matrix[0])

        # 1.1 start our "pointer" in the top-right (From EPI)
        row = 0
        col = width - 1
        while col >= 0 and row < height:
            if matrix[row][col] > target:
                col -= 1
            elif matrix[row][col] < target:
                row += 1
            else: # found it
                return True
        return False

    # Method 2: Sub optimal (Binary search in row/col by moving across diagonal)
    def binary_search(self, matrix, target, start, vertical):
        lo = start
        hi = len(matrix[0]) - 1 if vertical else len(matrix) - 1

        while hi >= lo:
            mid = (lo + hi) // 2
            if vertical: # searching a column
                if matrix[start][mid] < target:
                    lo = mid + 1
                elif matrix[start][mid] > target:
                    hi = mid - 1
                else:
                    return True
            else: # searching a row
                if matrix[mid][start] < target:
                    lo = mid + 1
                elif matrix[mid][start] > target:
                    hi = mid - 1
                else:
                    return True        
        return False

    def searchMatrix(self, matrix, target):
        # an empty matrix obviously does not contain `target`
        if not matrix:
            return False

        # iterate over matrix diagonals starting in bottom left.
        for i in range(min(len(matrix), len(matrix[0]))):
            vertical_found = self.binary_search(matrix, target, i, True)
            horizontal_found = self.binary_search(matrix, target, i, False)
            if vertical_found or horizontal_found:
                return True
        return False

    
    def findCircleNum_BFS(self, M: List[List[int]]) -> int:
        m = n = len(M)
        friends_circle = 0
        visited = [0 for i in range(m)]
        for city in range(m):
            if not visited[city]:
                friends_circle += 1
                que = [city]
                while que:
                    i = que.pop(0)
                    visited[i] = 1
                    for j in range(len(M)):
                        if M[i][j] == 1 and not visited[j]:
                            que.append(j)
        return friends_circle
    # Another way is to use the diagonal, rather than going to every element
    # For each diagonal element, go through its row and column and update the value in-place (no extra space)





################################################################################################################################
"""
Given a board and an end position for the player, write a function to determine if it is possible to travel from every open cell on the board to the given end position.

board1 = [
    [ 0,  0,  0, 0, -1 ],
    [ 0, -1, -1, 0,  0 ],
    [ 0,  0,  0, 0,  0 ],
    [ 0, -1,  0, 0,  0 ],
    [ 0,  0,  0, 0,  0 ],
    [ 0,  0,  0, 0,  0 ],
]

board2 = [
    [  0,  0,  0, 0, -1 ],
    [  0, -1, -1, 0,  0 ],
    [  0,  0,  0, 0,  0 ],
    [ -1, -1,  0, 0,  0 ],
    [  0, -1,  0, 0,  0 ],
    [  0, -1,  0, 0,  0 ],
]

board3 = [
    [ 0,  0,  0,  0,  0,  0, 0 ],
    [ 0, -1, -1, -1, -1, -1, 0 ],
    [ 0, -1,  0,  0,  0, -1, 0 ],
    [ 0, -1,  0,  0,  0, -1, 0 ],
    [ 0, -1,  0,  0,  0, -1, 0 ],
    [ 0, -1, -1, -1, -1, -1, 0 ],
    [ 0,  0,  0,  0,  0,  0, 0 ],
]

board4 = [
    [0,  0,  0,  0, 0],
    [0, -1, -1, -1, 0],
    [0, -1, -1, -1, 0],
    [0, -1, -1, -1, 0],
    [0,  0,  0,  0, 0],
]

board5 = [
    [0],
]

end1 = (0, 0)
end2 = (5, 0)

Expected output:

isReachable(board1, end1) -> True
isReachable(board1, end2) -> True
isReachable(board2, end1) -> False
isReachable(board2, end2) -> False
isReachable(board3, end1) -> False
isReachable(board4, end1) -> True
isReachable(board5, end1) -> True


n: width of the input board
m: height of the input board
"""
from collections import deque

# start param would be the end position of the given input
def bfs(board, start, end):
    queue, visited = deque(start), set() 
    while len(queue) > 0:
        cur_pos = queue.popleft()
        next_moves = findLegalMoves(board, cur_pos) 
        for pos in next_moves:
            if pos not in visited:
                queue.append(pos)
                visited.add(pos)
            if pos == end:
                return True 
    return False


def travelCheck(board, end):
    cache = [[False for j in range(len(board[0]))] for i in range(len(board))]
    for i in range(len(board)):
        for j in range(len(board[0])):
            if not board[i][j] == -1:
                start = (i,j)
                if not (dfs(board, set(), end, start, cache)):
                    return False
    return True
################################################################################################################################

def getRobotIndices(row):
    res = []
    for i in range(len(row)):
        if row[i] == 1:
            res.append(i)
    return res

def validPath(curPos, nextPos):
    if len(curPos) != len(nextPos):
        return False
    for i in range(len(curPos)):
        if abs(curPos[i] - nextPos[i] > 1):
            return False
    return True


# BFS in a weighted graph
# https://leetcode.com/problems/network-delay-time/
import heapq
class Solution(object):
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:        
        adj_list = defaultdict(list)
        
        for x,y,w in times:
            adj_list[x].append((w, y))
        
        visited=set()
        heap = [(0, k)]
        while heap:
            travel_time, node = heapq.heappop(heap)
            visited.add(node)
            
            if len(visited)==n: # optimization
                return travel_time
            
            for time, adjacent_node in adj_list[node]:
                if adjacent_node not in visited:
                    heapq.heappush(heap, (travel_time+time, adjacent_node))
                
        return -1


# https://leetcode.com/problems/maximal-square
# https://leetcode.com/problems/maximal-square/discuss/600149/Python-Thinking-Process-Diagrams-DP-Approach
# https://leetcode.com/problems/largest-plus-sign
# https://leetcode.com/problems/largest-plus-sign/solutions/1453636/intuitive-explained-with-image-short-and-clean-c/?orderBy=most_votes