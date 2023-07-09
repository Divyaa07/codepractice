from datetime import datetime
from functools import cache
import heapq
from logging import root
from readline import append_history_file


# https://leetcode.com/problems/reverse-substrings-between-each-pair-of-parentheses/
class Solution:
    def reverseParentheses(self, s: str) -> str:
        
        # Method 1 : O(n2)
        # stack = ['']
        # for c in s:
        #     # If ( is seen, start a new empty string in stack
        #     if c == '(':
        #         stack.append('')
        #     # If ) is seen, pop the last string in stack, reverse it and add it back
        #     elif c == ')':
        #         add = stack.pop()[::-1]
        #         stack[-1] += add
        #     # If char is seen, concatenate to the last string in stack
        #     else:
        #         stack[-1] += c
        # return stack.pop()

    
        # Method 2 : O(n)
        opened = []
        pair = {}
        # Find all pairs and save in a dictionary
        for i, c in enumerate(s):
            if c == '(':
                opened.append(i)
            if c == ')':
                j = opened.pop()
                pair[i], pair[j] = j, i

        # Traverse through the input string by switching directions with every () encountered
        # Try with (ed(et(oc))el) to understand
        res = []
        i, direction = 0, 1
        while i < len(s):
            if s[i] in '()':
                i = pair[i]
                direction = -direction
            else:
                res.append(s[i])
            i += direction
        return ''.join(res)


# https://leetcode.com/problems/single-number-ii
    def singleNumber(self, nums: List[int]) -> int:
        seen_once = seen_twice = 0
        
        for num in nums:
            # Explanation 1
            # first appearance: 
            # add num to seen_once 
            # don't add to seen_twice because of presence in seen_once
            
            # second appearance: 
            # remove num from seen_once 
            # add num to seen_twice
            
            # third appearance: 
            # don't add to seen_once because of presence in seen_twice
            # remove num from seen_twice


            # Explanation 2:
            # Accumulate the incoming number in ones provided twos is zero.
            # Twos will hold the number that has appeared twice.
            # If two becomes zero, it means the number has appeared the third time- Ones will hold that number now            
            seen_once = ~seen_twice & (seen_once ^ num)
            # Wait for ones bits to be zero before you increment twos.
            # Ones will be zero when the number is received twice.
            # So when the number will be received twice, we will store that in twos.            
            seen_twice = ~seen_once & (seen_twice ^ num)

        return seen_once


# Given a list of clock times, find the pair which has the smallest duration between them
# For instance, if given 07:42, 03:03, 02:15, and 10:23, the smallest duration is 48 minutes between the pair 02:15 and 03:03
# Your code should print the number of minutes between the times, as well as the two times in any order

# Methods: Sort O(nlogn), Brute Force: O(n2), DP: O(n)
# For instance, given the input above, your code should print 48, 02:15, 03:03

# list_of_times = ["17:21", "23:50", "04:01", "07:07", "03:03", "00:01", "12:42", "16:03"]
# list_of_times = ["07:42", "03:03", "02:15","10:23"]

def find_smallest_duration_optimized(times):
    # O(1) space, O(n) time
    # Idea: range of time in minutes is (00:00) to 23*60+59 = 0 to 1439

    if len(times) <= 1:
        return -1
    
    min_dur = float('inf')
    arr = [-1] * 1439
    for idx, time in enumerate(times):
        h, m = time.split(":")
        time_in_mins = int(h)*60 + int(m)
        arr[time_in_mins] = idx
    
    # Find two closest indices with values > -1 in arr: Linear scan of arr, O(n)
    last_index = None
    for idx, val in enumerate(arr):
        if val > -1:
            if last_index:
                min_dur = min(min_dur, idx - last_index)
            last_index = idx

    print(min_dur)
    return


# https://leetcode.com/problems/two-sum
def two_sum(nums, target):
    if not nums and len(nums) <= 1:
        return (-1, -1)
    d = {}
    for index, num in enumerate(nums):
        diff = target - num
        if diff in d:
            return (d[diff], index)
        d[num] = index
        return (-1, -1)

# Check string anagrams
def check_anagram1(str1, str2):
    str1 = str1.replace(' ', '')
    str2 = str2.replace(' ', '')
    if len(str1) != len(str2):
        return False
    anagram_check = {}

    for char in str1:
        if char in anagram_check:
            anagram_check[char] += 1
        else:
            anagram_check[char] = 1

    for char in str2:
        try:
            anagram_check[char] -= 1
        except KeyError:
            return False
    for key in anagram_check.keys():
        if key != 0:
            return False
    return True

def check_anagram2(str1, str2):

    def count_chars(str):
        count = {}
        for char in str:
            count[char] = count.get(char, 0) + 1
        return count

    str1 = str1.replace(' ', '')
    str2 = str2.replace(' ', '')
    if len(str1) != len(str2):
        return False
    return count_chars(str1) == count_chars(str2)


# https://leetcode.com/problems/group-anagrams
# Group anagrams
from collections import defaultdict
class Solution:
    # without sort, use array of 26 lowercase letters, optimize to O(nk) time, and O(nk) space.
    # map the tuple of array to string.    
    def groupAnagrams(strs):
        ans = defaultdict(list)
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            ans[tuple(count)].append(s)
        return ans.values()


# https://leetcode.com/problems/number-of-burgers-with-no-waste-of-ingredients
"""
4x + 2y = tomatoSlices
x + y = cheeseSlices
2x = tomatoSlices - 2 * cheeseSlices
x = (tomatoSlices - 2 * cheeseSlices) / 2
y = cheeseSlices - x
"""
def numOfBurgers(tomatoSlices, cheeseSlices):
    two_x = tomatoSlices - 2 * cheeseSlices
    x = two_x // 2
    y = cheeseSlices - x
    return [x, y] if two_x >= 0 and not two_x % 2 and y >= 0 else []


# https://leetcode.com/problems/license-key-formatting/
def licenseKeyFormatting(S, K):
    if not S:
        return S
    S = S.upper()
    S = ''.join(S.split('-'))
    dash_loc = len(S) - K
    while (dash_loc > 0):
        S = S[:dash_loc] + '-' + S[dash_loc:]
        dash_loc = dash_loc - K
    return S


# https://leetcode.com/problems/find-winner-on-a-tic-tac-toe-game/
def tictactoe(moves):
    # Winning Positions
    win = [[[0, 0], [1, 1], [2, 2]], [[2, 0], [1, 1], [0, 2]], [[0, 0], [0, 1], [0, 2]], [[1, 0], [1, 1], [1, 2]],
           [[2, 0], [2, 1], [2, 2]], [[0, 0], [1, 0], [2, 0]], [[0, 1], [1, 1], [2, 1]], [[0, 2], [1, 2], [2, 2]]]

    # Check if current positions will win
    def checkWin(arr):
        for i in win:
            if all(x in arr for x in i):
                return True
        return False

    # Capture moves of A and B separately
    A = []  # Moves of A
    B = []  # Moves of B
    n = len(moves)
    for i in range(n):
        if i % 2 == 0:
            A.append(moves[i])
        else:
            B.append(moves[i])

    # A wins
    if (checkWin(A)):
        return "A"
    # B wins
    elif (checkWin(B)):
        return "B"
    # None of them won and the board is full => Result is Draw
    elif n == 9:
        return "Draw"
    # None of them won and the board is not full => Result is Pending
    else:
        return "Pending"


# https://leetcode.com/problems/bulls-and-cows/
def bullsCows(secret, guess):
    bull, cow = 0, 0
    s = {} # secret hashtable
    g = {} # guess hashtable

    for i in xrange(len(secret)):
        if secret[i] == guess[i]:
            bull += 1
        else:
            s[secret[i]] = s.get(secret[i], 0) + 1
            g[guess[i]] = g.get(guess[i], 0) + 1

    for k in s:
        if k in g:
            cow += min(s[k], g[k])

    return '{0}A{1}B'.format(bull, cow)

def bullsCows(secret, guess):
    s, g = Counter(secret), Counter(guess)
    a = sum(i == j for i, j in zip(secret, guess))
    return '%sA%sB' % (a, sum((s & g).values()) - a)


# https://leetcode.com/problems/isomorphic-strings
# https://leetcode.com/problems/find-and-replace-pattern
def isoMorphicStrings(s1, s2):
    if len(s1) != len(s2):
        return False
    s1tos2 = {}
    s2tos1 = {}
    for i in range(s1):
        if s1[i] in s1tos2 and s2[i] in s2tos1: # has to be present in both
            if s1[i] != s2tos1[s2[i]] or s2[i] != s1tos2[s1[i]] # apple and zaapy, paper and title => true
                return False
        elif s1[i] not in s2 or s2[i] not in s1: # false, if not present in either one of the input string
            return False
        else: # add it to the hashmap
            s1tos2[s1[i]] = s2[i]
            s2tos1[s2[i]] = s1[i]
    return True

def findAndReplacePattern(words, pattern):
    isomorphicWords = []
    for w in words:
        if isoMorphicStrings(w, pattern):
            isomorphicWords.append(w)
    return isIsomorphicWords

def isIsomorphic1(s, t):
    d1, d2 = {}, {}
    for i, val in enumerate(s):
        d1[val] = d1.get(val, []) + [i]
    for i, val in enumerate(t):
        d2[val] = d2.get(val, []) + [i]
    return sorted(d1.values()) == sorted(d2.values())

def isIsomorphic2(self, s, t):
    # Both strings should have same set of chars to swap
    d1, d2 = dict(), dict()
    for v, w in zip(s,t):
        if (v in d1 and d1[v] != w) or (w in d2 and d2[w] != v):
                return False
        d1[v], d2[w] = w, v
    return True

def isIsomorphic3(s, t):
    # Check if same set of chars are found at the same position
    m1, m2, n = {}, {}, len(s)
    for i in range(n):
        if (m1[s[i]] != m2[t[i]]):
            return false
        m1[s[i]] = i + 1
        m2[t[i]] = i + 1
    return true


# https://leetcode.com/problems/min-cost-climbing-stairs
def func(cost):
    pre, cur = cost[0], cost[1]
    for i in range(2, len(cost)):
        tmp = cur
        cur = min(pre, cur) + cost[i]
        pre = tmp
        # pre, cur = cur, min(pre, cur) + cost[i]
    return min(pre,cur)



# https://leetcode.com/problems/contains-duplicate-iii/
# Similar to bucket sort solution (Putting input elements in buckets)
def containsNearbyAlmostDuplicate(self, nums, k, t):
    if t < 0 or k < 0:
        return False
    allBuckets = {}
    bucketSize = t + 1 # t=0 case will be handled automatically
    for i in range(len(nums)):
        # m is bucket Index for nums[i]
        m = nums[i] // bucketSize

        # if there is a bucket already present corresponding to current number
        if m in allBuckets:
            return True

        # checking two adjacent buckets  m, m-1
        if (m - 1) in allBuckets and abs(nums[i] - allBuckets[m - 1]) < bucketSize:
            return True

        # checking two adjacent buckets m, m+1
        if (m + 1) in allBuckets and abs(nums[i] - allBuckets[m + 1]) < bucketSize:
            return True
        allBuckets[m] = nums[i]

        # removing the bucket corresponding to number out of our k sized window
        if i >= k:
            del allBuckets[nums[i - k] // bucketSize]
    return False


# https://leetcode.com/problems/maximum-profit-in-job-scheduling
class Solution:
    def jobScheduling(self, startTime, endTime, profit):
        jobs = sorted(zip(startTime, endTime, profit), key=lambda v: v[0])
        hp = []
        total = 0

        for s,e,p in jobs:
            while hp and hp[0][0] <= s:
                popd = heappop(hp)
                total = max(total, popd[1])

            heappush(hp, (e, p + total))

        while hp:
            popd = heappop(hp)
            total = max(total, popd[1])

        return total

# https://leetcode.com/problems/strobogrammatic-number/
# A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down)
class Solution:
    def isStrobogrammatic1(self, num: str) -> bool:
        d = {'6': '9',
             '9': '6',
             '0': '0',
             '8': '8',
             '1': '1'}
        rotated = ''
        for i in range(len(num)):
            if num[i] in d:
                rotated += d[num[i]]
            else:
                return False
        return rotated[::-1] == num

    def isStrobogrammatic2(self, num: str) -> bool:
        d = (('6', '9'),
             ('9', '6'),
             ('0', '0'),
             ('8', '8'),
             ('1', '1'))

        for i in range(len(num) + 1 // 2):
            if (num[i], num[~i]) not in d:
                return False
        return True


# https://leetcode.com/problems/linked-list-random-node/

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
import random
class Solution:

    def __init__(self, head: ListNode):
        """
        @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node.
        """
        self.head = head
        self.reservoir = []
        self.start = 0
        self.pool_size = 1
        self.node = head

        if not head:
            return

        while self.start < self.pool_size and self.node.next:
            self.reservoir.append(self.node.val)
            self.node = self.node.next
            self.start += 1

    def getRandom(self) -> int:
        """
        Returns a random node's value based on reservoir pool algorithm
        https://en.wikipedia.org/wiki/Reservoir_sampling
        """
        if not self.head:
            return
        if not self.head.next:
            return self.head.val
        res = [self.head.val]
        next_node = self.head.next
        nodes = 2
        while (next_node):
            j = random.randrange(1, nodes + 1)
            if j <= self.pool_size:
                res[j - 1] = next_node.val
            next_node = next_node.next

        return res[0] if self.pool_size == 1 else res

# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()

# https://leetcode.com/problems/3sum
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        l = len(nums)

        def threesum_helper(i, l):
            lo = i + 1
            high = l - 1

            while lo < high:
                sum = nums[lo] + nums[high] + nums[i]
                if sum < 0 or (lo > i + 1 and nums[lo] == nums[lo - 1]):
                    lo += 1
                elif sum > 0 or (high < l - 1 and nums[high] == nums[high + 1]):
                    high -= 1
                else:
                    res.append([nums[i], nums[lo], nums[high]])
                    lo += 1
                    high -= 1

        for i in range(len(nums) - 1):
            if nums[i] > 0:  # Rest of values cannot sum to 0 since it is sorted!
                break

            if i == 0 or nums[i] != nums[i - 1]:
                threesum_helper(i, l)

        return res

# https://leetcode.com/problems/3sum-smaller/
# https://leetcode.com/problems/3sum-with-multiplicity/
class Solution:
    def threeSumSmaller(self, nums: List[int], target: int) -> int:
        nums.sort()
        res = 0
        l = len(nums)

        def threesum_helper(i, l, target):
            lo = i + 1
            high = l - 1
            result = 0

            while lo < high:
                sum = nums[i] + nums[lo] + nums[high]
                if sum < target:
                    result += (high - lo)
                    lo += 1
                else:
                    high -= 1
            return result

        for i in range(len(nums) - 1):
            res += threesum_helper(i, l, target)

        return res

    # https://leetcode.com/problems/3sum-closest
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        diff = float('inf')
        res = float('inf')
        nums.sort()

        for i in range(len(nums)):
            lo = i + 1
            hi = len(nums) - 1
            while lo < high:
                s = nums[i] + nums[lo] + nums[hi]
                if abs(s - target) < diff:
                    diff = abs(s - target)
                    res = s

                if s < target:
                    lo += 1
                else:
                    hi -= 1
            # Return if you find the best possible answer
            if diff == 0:
                break

        return res

# Good problem for learning backtracking method
# https://leetcode.com/problems/swap-nodes-in-pairs/
# https://leetcode.com/articles/swap-nodes-in-pairs/
# https://leetcode.com/problems/target-sum


# DP solution
# https://leetcode.com/problems/maximum-subarray
def find_maximum_sum_subarray(A):
    min_sum=max_sum=0
    for running_sum in itertools.accumulate (A):
        min_sum = min(min_sum, running_sum)
        max_sum = max(max_sum, running_sum - min_sum)
    return max-sum

def reverse_digit(n):
    res = 0
    while n:
        first_digit = n % 10
        n = n // 10
        res = res * 10 + first_digit
    return res

# https://leetcode.com/problems/missing-element-in-sorted-array
def binary_search(nums):
        if not nums or k == 0:
        return 0
        
        diff = nums[-1] - nums[0] + 1 # complete length
        missing = diff - len(nums) # complete length - real length
        if k > missing: # if k is larger than the number of mssing words in sequence
            return nums[-1] + k - missing
        
        left, right = 0, len(nums) - 1
        while left + 1 < right:
            mid = (left + right) // 2
            missing = nums[mid] - nums[left] - (mid - left)
            if missing < k:
                left = mid
                k -= missing # KEY: move left forward, we need to minus the missing words of this range
            else:
                right = mid
                
        return nums[left] + k # k should be between left and right index in the end

def iterative_approach(nums):
    for i in range(1, len(nums):
        diff = nums[i]-nums[i-1]-1
        if (diff>=k):
            return nums[i-1]+k
        k -= diff

    return nums[n-1]+k


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