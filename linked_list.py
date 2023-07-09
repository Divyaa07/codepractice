import heapq
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# https://leetcode.com/problems/add-two-numbers/
# https://leetcode.com/problems/add-binary/
# (Similar solution for both problems)
# Add two numbers with each digit in a linked list
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
#         # 1. Recursive (O(n) time and space)
#         if not head or not head.next:
#             return head

#         # temp will be the new header passed to previous call stacks
#         temp = self.reverseList(head.next)
#         head.next.next = head
#         head.next = None
#         return temp
    
        # 2. Iterative solution
        prev = None
        current = head
        while current:
            temp = current.next
            current.next = prev
            prev = current
            current = temp
        return prev
    
    def add_two_linked_lists_numbers(l1, l2):
        """
        Follow UP: Numbers in non-reversed order, use below reverse method
        """

        start = ListNode(0)
        cur = start
        carry = 0
        while (l1 or l2 or carry):
            l1_val = l1.val if l1 else 0
            l2_val = l2.val if l2 else 0
            tmp_num = l1_val + l2_val + carry
            carry, num = tmp_num // 10, tmp_num % 10 # In case of binary, divide by 2 instead
            cur.next = ListNode(num)
            cur, l1, l2 = cur.next, l1.next if l1 else None, l2.next if l2 else None
        return start.next


# LRU cache
# https://leetcode.com/problems/lru-cache/
class DLinkedNode(): 
    def __init__(self):
        self.key = 0
        self.value = 0
        self.prev = None
        self.next = None
            
class LRUCache():

    def __init__(self, capacity):
        """
        :type capacity: int
        head -> node1 -> node2 -> nodeN -> tail
        """
        self.cache = {}
        self.size = 0
        self.capacity = capacity
        # Maintain a dummy head and tail node to avoid many corner cases!
        self.head, self.tail = DLinkedNode(), DLinkedNode()

        self.head.next = self.tail
        self.tail.prev = self.head    
    
    def _add_node(self, node):
        """
        Always add the new node right after head.
        """
        node.prev = self.head
        node.next = self.head.next

        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        """
        Remove an existing node from the linked list.
        """
        prev = node.prev
        new = node.next

        prev.next = new
        new.prev = prev

    def _move_to_head(self, node):
        """
        Move certain node in between to the head.
        """
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self):
        """
        Pop the current tail.
        """
        res = self.tail.prev
        self._remove_node(res)
        return res
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        node = self.cache.get(key, None)
        if not node:
            return -1
        # move the accessed node to the head;
        self._move_to_head(node)
        return node.value

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: Non
        """
        node = self.cache.get(key, None)
        if not node:
            newNode = DLinkedNode()
            newNode.key = key
            newNode.value = value

            self.cache[key] = newNode
            self._add_node(newNode)
            self.size += 1

            if self.size > self.capacity:
                # pop the tail
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.size -= 1
        else:
            # update the value.
            node.value = value
            self._move_to_head(node)
