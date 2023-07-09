class Node():
    def __init__(self, val):
        self.left = None
        self.right = None
        self.data = val


root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)

########################################################################################################################
## TREE TRAVERSAL BASICS
########################################################################################################################
# Tree Traversals
# Depth first traversals
def inorder_traversal(obj):
    if not obj:
        return
    inorder_traversal(obj.left)
    print(obj.data)
    inorder_traversal(obj.right)

# Iterative traversal
def inorderTraversal1(root):
    in_order = []
    stack, p = [], root
    while stack or p:
        if p:
            stack.append(p)
            p = p.left
        else:
            p = stack.pop() # Pops from right end of the list
            in_order.append(p.val)
            p = p.right
    return in_order

def inorderTraversal2(root):
    in_order = []
    stack, p = [], root
    while stack or p:
        while p:
            stack.append(p)
            p = p.left

        p = stack.pop() # Pops from right end of the list
        in_order.append(p.val)
        p = p.right
    return in_order


def preorder_traversal(obj):
    if not obj:
        return
    print(obj.data)
    preorder_traversal(obj.left)
    preorder_traversal(obj.right)


def postorder_traversal(obj):
    if not obj:
        return
    postorder_traversal(obj.left)
    postorder_traversal(obj.right)
    print(obj.data)


# Binary tree operations
# Lookup
def lookup_binary_tree(node, target):
    if not node.data:
        return False
    if target == node.data:
        return True

    if target < node.data:
        return lookup_binary_tree(node.left, target)
    else:
        return lookup_binary_tree(node.right, target)

# Insert into binary tree
def insert_binary_tree(node, data):
    if not node.data:
        return Node(data)

    if data <= node.data:
        node.left = insert_binary_tree(node.left, data)
    else:
        node.right = insert_binary_tree(node.right, data)

# Delete a node from a BST
# https://leetcode.com/problems/delete-node-in-a-bst/solution/
def delete_bst_node(value):
    pass

# Size of binary tree
def size_binary_tree(root):
    if not root:
        return 0
    return count_nodes_1(root, 0)
def count_nodes_1(node, count):
    if not node:
        return 0
    left_tree_count = count_nodes_1(node.left, count+1)
    right_tree_count = count_nodes_1(node.right, count+1)
    return left_tree_count + right_tree_count + 1

def count_nodes_2(node):
    if not node:
        return 0
    return count_nodes_2(node.left) + count_nodes_2(node.right) + 1

# Depth of a tree
def depth_tree(root):
    if not root:
        return 0
    lheight = depth_tree(root.left)
    rheight = depth_tree(root.right)
    return lheight+1 if lheight > rheight else rheight+1

# Min value in a binary search tree
def min_tree(root):
    if not root.left:
        return root.data
    return min_tree(root.left)

# Think how would you find successor/predecessor of a node in a binary tree
# (required for deleting a node in a binary search tree)

# Has path sum in a binary tree
def has_path_sum(root, sum):
    if not root:
        return sum==0
    return has_path_sum(root.left, sum - root.data) | \
           has_path_sum(root.right, sum - root.data)

# Print all paths to leaf in a binary tree
# Simple example of Backtracking !!
def print_path(root, path=[]):
    if not root:
        return
    path.append(root.data)
    if not root.left and not root.right:
        print(path)
        path.pop()
        return
    print_path(root.left, path)
    print_path(root.right, path)
    path.pop() # this is important

# Do a mirror of given binary tree
def mirror_binary_tree(node):
    if not node:
        return
    mirror_binary_tree(node.left)
    mirror_binary_tree(node.right)
    temp = node.left
    node.left = node.right
    node.right = temp

#Check if it is a valid Binary Search Tree (Using NULL instead of INT_MIN or INT_MAX
def check_bst(root):
    if root:
        return is_bst_check(root, None, None)
    return False
def is_bst_check(node, min, max):
    if not node:
        return True
    if min and node.data < min:
        return False
    if max and node.data > max:
        return False
    return is_bst_check(node.left, min, node.val) & is_bst_check(node.right, node.val, max)

# Valid Binary search tree (DFS concept with stack)
def is_binary_search_tree(root):
    # Start at the root, with an arbitrarily low lower bound
    # and an arbitrarily high upper bound
    node_and_bounds_stack = [(root, -float('inf'), float('inf'))]
    # Depth-first traversal
    while len(node_and_bounds_stack):
        node, lower_bound, upper_bound = node_and_bounds_stack.pop()
        # If this node is invalid, we return false right away
        if (node.value <= lower_bound) or (node.value >= upper_bound):
            return False
        if node.left:
            # This node must be less than the current node
            node_and_bounds_stack.append((node.left, lower_bound, node.value))
        if node.right:
            # This node must be greater than the current node
            node_and_bounds_stack.append((node.right, node.value, upper_bound))

    # If none of the nodes were invalid, return true
    # (at this point we have checked all nodes)
    return True


# https://leetcode.com/problems/inorder-successor-in-bst
def inorderSuccessor(root, p):
    successor = None
    # case 1. turn right, then left..left
    if p.right:
        node = p.right
        while node:
            successor = node
            node = node.left
    # case 2. ordinary search in bst
    else:
        node = root
        while node:
            if node.val == p.val:
                break
            if p.val > node.val:
                node = node.right
            else:
                successor = node # any root node, where root.left=..p..
                node = node.left
    return successor


# Binary search
def binary_search_array(arr, x):
    def bs_array_recursive(arr, l, r, elem):
        if l > r:
            return -1
        mid = l + (r-l)//2
        if arr[mid] == elem:
            return mid
        elif arr[mid] < elem:
            return bs_array_recursive(arr, mid+1, r, elem)
        else:
            return bs_array_recursive(arr, l, mid-1, elem)

    def bs_array_iterative(arr, x):
        left, right = 0, len(arr)-1
        while left < right:
            mid = left + (right - left) / 2
            if arr[mid] == x:
                return mid
            if arr[mid] > x:
                right = mid-1
            else:
                left = mid + 1
        return -1

    return bs_array_recursive(arr, 0, len(arr)-1, x)


# Return second largest element from a BST
# For a tree like,
#    137
# 42
#    99
# The answer is in the left subtree

def find_second_largest_bst_element_iterative(root):

    def find_largest(root):
        current = root
        while current:
            if not current.right:
                return current.value
            current = current.right

    cur_node = root
    while cur_node:
        if cur_node.right:
            # If cur node has no grandchildren, this is second largest
            if not cur_node.right.right and not cur_node.right.left:
                return cur_node
            # Any grandchildren found, move to its right
            cur_node = cur_node.right

        # root is largest, second largest in left of root
        else:
            return find_largest(cur_node.left)

def find_second_largest_bst_element_recursive(root):

    def find_largest(root):
        current = root
        while current:
            if not current.right:
                return current.value
            current = current.right

    if root.right:
        # If cur node has no grandchildren, this is second largest
        if not root.right.right and not root.right.left:
            return root
        # If cur node has any grandchildren, go further right
        return find_second_largest_bst_element_recursive(root.right)
    # root is largest, second largest in left of root
    else:
        return find_largest(root.left)


# https://leetcode.com/problems/count-complete-tree-nodes
# In a complete binary tree, each level l would have 2^l nodes at max
# With this logic, if the tree has uneven levels, 
# then 2^(depth of right tree) + count_nodes(left tree) is the answer
def countNodes(root):
    if not root:
        return 0
    leftDepth = getDepth(root.left)
    rightDepth = getDepth(root.right)
    if leftDepth == rightDepth:
        return pow(2, leftDepth) + countNodes(root.right)
    else:
        return pow(2, rightDepth) + countNodes(root.left)

def getDepth(root):
    if not root:
        return 0
    return 1 + getDepth(root.left)


# https://leetcode.com/problems/serialize-and-deserialize-binary-tree
def serialize(self, root):
    if not root: return 'x'
    return root.val, self.serialize(root.left), self.serialize(root.right)

def deserialize(self, data):
    if data[0] == 'x': return None
    node = Node(data[0])
    node.left = self.deserialize(data[1])
    node.right = self.deserialize(data[2])
    return node

# AVL tree insertion
# The AVL tree and other self-balancing search trees like Red Black are useful to get all basic operations done in O(log n) time. 
# The AVL trees are more balanced compared to Red-Black Trees, but they may cause more rotations during insertion and deletion. 
# So if your application involves many frequent insertions and deletions, then Red Black trees should be preferred. 
# And if the insertions and deletions are less frequent and search is the more frequent operation, then AVL tree should be preferred
# https://www.geeksforgeeks.org/avl-tree-set-1-insertion/


# https://leetcode.com/problems/closest-leaf-in-a-binary-tree/
"""
Given a binary tree where every node has a unique value, and a target key k, find the value of the nearest leaf node to target k in the tree.
Here, nearest to a leaf means the least number of edges travelled on the binary tree to reach any leaf of the tree. Also, a node is called a leaf if it has no children.
In the following examples, the input tree is represented in flattened form row by row. The actual root tree given will be a TreeNode object.

Input:
root = [1,2,3,4,null,null,null,5,null,6], k = 2
Diagram of binary tree:
             1
            / \
           2   3
          /
         4
        /
       5
      /
     6

Output: 3
Explanation: The leaf node with value 3 (and not the leaf node with value 6) is nearest to the node with value 2.

Ans: BFS on undirected graph
"""
class Solution(object):
    def findClosestLeaf(self, root, k):
        # Time: O(n)
        # Space: O(n)
        from collections import defaultdict
        graph, leaves = defaultdict(list), set()
        # Graph construction
        def traverse(node):
            if not node:
                return
            if not node.left and not node.right:
                leaves.add(node.val)
                return
            if node.left:
                graph[node.val].append(node.left.val)
                graph[node.left.val].append(node.val)
                traverse(node.left)
            if node.right:
                graph[node.val].append(node.right.val)
                graph[node.right.val].append(node.val)
                traverse(node.right)
        traverse(root)

        # If a node has only one edge in an undirected graph, it is a leaf node!
        # if len(graph[node]) <= 1:
        #     return node.val
        # Graph traversal - BFS
        queue = [k]
        while len(queue):
            next_queue = []
            for node in queue:
                if node in leaves:
                    return node
                next_queue += graph.pop(node, [])
            queue = next_queue


# https://leetcode.com/problems/binary-tree-vertical-order-traversal
from collections import defaultdict
class Solution:
    def verticalOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []

        columnTable = defaultdict(list)
        min_column = max_column = 0
        queue = deque([(root, 0)])

        while queue:
            node, column = queue.popleft()

            if node is not None:
                columnTable[column].append(node.val)
                min_column = min(min_column, column)
                max_column = max(max_column, column)

                queue.append((node.left, column - 1))
                queue.append((node.right, column + 1))

        return [columnTable[x] for x in range(min_column, max_column + 1)]


# Comparing sum of left and right branch of a binary tree from array representation
# Suppose you are giving a binary tree represented as an array. For example, [3, 6, 2, 9, -1, 10] retpresents the following binary tree, where -1 indicates it is a NULL node.
# Input: [3,6,2,9,-1,10] should return left
def compare_left_right_binary_tree(arr):
    def sum(index):
        if index < len(arr):
            if arr[index] == -1:
                return 0
            return arr[index] + sum(index*2 + 1) + sum(index*2+2)
        else:
            return 0
        
    left = sum(1)
    right = sum(2)
    if left > right:
        return "Left"
    elif right > left:
        return "Right"
    else:
        return ""