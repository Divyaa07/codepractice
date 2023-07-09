
# Some tips
"""
Bit manipulation:
    - Given an integer x, remove the lowest set bit (x & x-1) of x
    - Given an integer x, extract the lowest set bit (x & ~(x-1)) of x
    - XOR has associative property

Given an integer:
    - Get number of integer digits: math.log(number, base)
    - MSD Mask: math.floor(log(number, base)-1)
    - Get the LSD: number % base
    - Get the MSD: number % msd_mask
"""


# Using XOR for problems (similar concept)
# Find one single element in an array of all duplicates
# Find two Missing Numbers in array of size (n-2) with range (1,n) [use last bit set]
# Two repeating elements in array of size (n+2) with range (1,n) [use last bit set]
# Two odd occuring elements in array while others occur are even times


# Sort an almost sorted array ([1,2,6,4,5,3,7])
def find_swapped_elems_in_sorted_array(nums):
    n = len(nums)
    x = y = -1
    for i in range(n - 1):
        if nums[i + 1] < nums[i]:
            y = nums[i + 1]
            # first swap occurence
            if x == -1:     
                x = nums[i]
            # second swap occurence, so swap with already found anomaly element
            else:           
                break
    return x, y


# https://leetcode.com/problems/product-of-array-except-self
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # Method 1: Calculate left and right products in two traversals
        # Keep default as 1 on left and right sides
        answer = [1]*len(nums)
        leftProduct = 1
        for i in range(0, len(nums)):
            answer[i] = answer[i] * leftProduct
            leftProduct *= nums[i]
        rightProduct = 1
        for i in range(len(nums)-1, -1, -1):
            answer[i] = answer[i] * rightProduct
            rightProduct *= nums[i]
        return answer

        # Method 2: In Place solution in O(1) space! (Using division is not that easy!)
        product = 1
        zero = 0
        for num in nums:
            if num == 0:
                zero += 1
            else:
                product *= num
        for i in range(0, len(nums)):
            if zero == 0:
                nums[i] = product // nums[i]
            elif zero == 1 and nums[i] == 0:
                nums[i] = product
            else:
                nums[i] = 0
        return nums


# Reverse a string recursively (do not print, but return as a char)
def reverse_string(str):
    """Input is a python string"""
    if not str:
        return str
    return ''.join(recurse_reverse(str, 0, []))
def recurse_reverse(str, pos, rstr):
    if pos == len(str):
        return
    recurse_reverse(str, pos+1, rstr)
    rstr.append(str[pos])
    if pos == 0:
        return rstr

def reverse_string_in_place(str):
    """str is a python string"""
    if str == "":
        return str
    else:
        return reverse_string_in_place(str[1:]) + str[0]

def reverse_string_list(str):
    """Input is a python string"""
    if not str:
        return str
    return recurse_reverse_list(str, len(str), 0)
def recurse_reverse_list(str, len_str, pos):
    if pos == int(len_str/2):
        return
    recurse_reverse_list(str, len_str, pos+1)
    temp = str[pos]
    str[pos] = str[len_str-pos-1]
    str[len_str-pos-1] = temp
    if pos == 0:
        return str


# Merge two sorted lists
def merge_lists(l1, l2):
    len1, len2 = len(l1), len(l2)
    temp, i, j = [], 0, 0
    while((i < len1) & (j < len2)):
        if l1[i] == l2[j]:
            temp.extend([l1[i], l2[j]])
            i += 1
            j += 1
        elif l1[i] > l2[j]:
            temp.append(l2[j])
            j += 1
        elif l2[j] > l1[i]:
            temp.append(l1[i])
            i += 1
    if i==len1:
        temp.extend(l2[j:])
    else:
        temp.extend(l1[i:])
    return temp

# Merge several sorted lists
def merge_several_lists(input_lists):
    result = merge_lists(input_lists[0], input_lists[1])
    for i in range(2, len(input_lists), 2):
        tmp_result = merge_lists(input_lists[i], input_lists[i+1])
        result.extend(tmp_result)
    return result


# Letter combinations in phone keypad (recursive, iterative)
# https://leetcode.com/problems/letter-combinations-of-a-phone-number
def letterCombinations(digits):
    if not digits:
        return []
    keypad = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno',
              '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
    result = ['']
    for num in digits:
        temp = []
        for i in result:
            for l in keypad[num]:
                temp.append(i + l)
        result = temp
    return result

def letterCombinations(digits):
    if not digits:
        return []
    result = ['']
    keypad = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno',
              '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
    for i in digits:
        result = [prev + char for prev in result for char in keypad[i]]
    return result

# Find all subsets of a set (with no duplicates)
# Iterative
def find_subsets(nums):
    res = [[]]
    for num in nums:
        res += [i + num for i in res]
    return res

# Recursive
def dinstinct_subsets(nums):
    res = []
    def subsets(nums, path=[]):
       res.append(path)
       for i in range(len(nums)):
           subsets(nums[i+1:], path+[nums[i]])
       return res
    return subsets(nums)

# GCD of N numbers
def gcd1(a, b):
    if a==0:
        return b
    gcd1(b%a, a)

def gcd2(a, b):
    if b==0:
        return b
    gcd2(b, a%b)

def gcd_list(nums):
    result = nums[0]
    for i in range(1, len(nums)):
        result = gcd1(i, result)
    return result

def remove_all_list_duplicates_in_place(nums):
    if not nums:
        return None
    l = len(nums)
    if len(nums) == 1:
        return nums
    p1, p2 = 0, 1

    while (p2 < l):
        if nums[p1] == nums[p2]:
            p2 += 1
        else:
            nums[p1 + 1] = nums[p2]
            p1 += 1
            p2 += 1
    del nums[p1 + 1:p2]


# Reverse words in a list/string
# https://leetcode.com/problems/reverse-words-in-a-string/
# https://leetcode.com/problems/reverse-string-ii/
# https://leetcode.com/problems/reverse-words-in-a-string-iii/


# Pascal's triange


# Escape from large maze
# https://leetcode.com/problems/escape-a-large-maze/


# Zigzag conversion
# https://leetcode.com/problems/zigzag-conversion/

# 2nd largest element in binary search tree
# https://www.interviewcake.com/question/python3/second-largest-item-in-bst?utm_source=weekly_email&utm_source=drip&utm_campaign=weekly_email&utm_campaign=Interview%20Cake%20Weekly%20Problem%20%23252:%202nd%20Largest%20Item%20in%20a%20Binary%20Search%20Tree&utm_medium=email&utm_medium=email

# Print concentric rectangular pattern in a 2d matrix
def prettyPrint(A):
    """
    For input of 4,
    4 4 4 4 4 4 4
    4 3 3 3 3 3 4
    4 3 2 2 2 3 4
    4 3 2 1 2 3 4
    4 3 2 2 2 3 4
    4 3 3 3 3 3 4
    4 4 4 4 4 4 4
    """
    res = [[A] * ((A * 2) - 1)]
    l1 = len(res[0])
    a = A - 1
    for i in range(1, A):
        tmp_list = res[i - 1][:]
        # tmp_list[i:l1-i-2] = a * (l1-i-2)
        for j in range(i, l1 - i):
            tmp_list[j] = a
        res.append(tmp_list)
        a -= 1
    for i in range(1, A):
        res.append(res[A - 1 - i])
    return res

def prettyPrint(A):
    return [[1 + max(abs(i), abs(j)) for j in range(-A + 1, A)] for i in range(-A + 1, A)]


def my_multiply_two_strings(n1, n2):
    res = [0 * (len(n1) + len(n2))]
    pos = len(res) - 1
    for i in n1[::-1]:
        tmp_pos = pos
        for j in n2[::-1]:
            res[tmp_pos] += (ord(n2[j] - ord('0') * ord(n1[i] - ord['0'])))
            res[tmp_pos-1] += res[tmp_pos] // 10
            res[tmp_pos] %= 10
            tmp_pos -= 1
        pos -= 1

    # Remove the leading zeroes in the initialized array
    zeroes_ptr = 0
    for i in res:
        if res[i] == 0:
            zeroes_ptr += 1
        else:
            break
    return ''.join(res[zeroes_ptr:])

# https://leetcode.com/problems/multiply-strings/
def multiple_two_string_numbers(num1, num2):
    product = [0] * (len(num1) + len(num2)) #placeholder for multiplication ndigit by mdigit result in n+m digits
    position = len(product)-1 # position within the placeholder

    for n1 in num1[::-1]:
        tempPos = position
        for n2 in num2[::-1]:
            product[tempPos] += int(n1) * int(n2) # adding the results of single multiplication
            product[tempPos-1] += product[tempPos]//10 # bring out carry number to the left array
            product[tempPos] %= 10 # remove the carry out from the current array
            tempPos -= 1 # first shifting the multplication to the end of the first integer
        position -= 1 # then once first integer is exhausted shifting the second integer and starting

    # once the second integer is exhausted we want to make sure we are not zero padding
    pointer = 0 # pointer moves through the digit array and locate where the zero padding finishes
    while pointer < len(product)-1 and product[pointer] == 0: # if we have zero before the numbers shift the pointer to the right
        pointer += 1

    return ''.join(map(str, product[pointer:])) # only report the digits to the right side of the pointer

def multiply_nums_new_implementation(num1, num2):
    res = [0] * (len(num1) + len(num2) + 1)
    for i, val in enumerate(num1[::-1]):
        tmp_pos = i
        for j in num2[::-1]:
            res[tmp_pos] += (val * j)
            res[tmp_pos+1] += (res[tmp_pos] // 10)
            res[tmp_pos] = res[tmp_pos] % 10
            tmp_pos += 1

    tmp_pos = len(res)-1
    while tmp_pos > 0 and res[tmp_pos] == 0:
        tmp_pos -= 1
    return res[:tmp_pos+1][::-1]


# https://leetcode.com/problems/longest-palindromic-substring/
def longest_palindrome(s):
    res = ''
    input_len = len(s)

    def find_longest_palindrome(l, r):
        while l >= 0 and l < input_len and r >= 0 and r < input_len and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l + 1:r]

    for i in range(input_len):
        tmp1, tmp2 = find_longest_palindrome(i, i), find_longest_palindrome(i, i + 1)
        tmp_res = tmp1 if len(tmp1) > len(tmp2) else tmp2
        if len(tmp_res) > len(res):
            res = tmp_res
    return res

# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
def maxProfit_without_negative_profit(self, prices):
    # Returns 0 if no positive profit can be made
    max_profit, least_price = 0, float("inf")
    for i, price in enumerate(prices):
        if prices[i] <= least_price:
            least_price = prices[i]
        else:
            cur_profit = prices[i] - least_price
            max_profit = max(cur_profit, max_profit)
    return max_profit

def maxProfit_with_negative_profit(self, prices):
    # Returns profit, even if its negative
    if len(prices) < 2:
        return 0
    min_price  = prices[0]
    max_profit = prices[1] - prices[0]
    for index in range(1, len(prices)):
        potential_profit = prices[index] - min_price
        max_profit = max(max_profit, potential_profit)
        min_price = min(min_price, prices[index])
    return max_profit

# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
# Complete as many transactions as you like
def maxProfit_with_sell_buy_not_allowed_on_same_day(prices):
    # Buy at valley, sell at peak!
    i, buy, sell, profit, N = 0, 0, 0, 0, len(prices)
    while i < N:
        # Find the valley
        while (i < N and prices[i + 1] <= prices[i]):
            i += 1
        buy = prices[i]
        # Find the peak
        while (i < N and prices[i + 1] > prices[i])
            i += 1
        sell = prices[i]

        profit += sell - buy

    return profit

def maxProfit_with_buy_sell_allowed_on_same_day_1(prices):
    # Returns profit, even if its negative
    max_profit, least_price = 0, float("inf")
    for i, price in enumerate(prices):
        if prices[i] <= least_price:
            least_price = prices[i]
        else:
            cur_profit = prices[i] - least_price
            if cur_profit > 0:
                max_profit = max_profit + cur_profit
            least_price = prices[i]
    return max_profit

def maxProfit_with_buy_sell_allowed_on_same_day_2(prices):
    maxprofit = 0;
    for i in range(1, prices):
        if (prices[i] > prices[i - 1]):
            maxprofit += prices[i] - prices[i - 1]
    return maxprofit


# Max sum subarray of size K [SLIDING WINDOW PATTERN]
def max_sub_array_of_size_k(k, arr):
    max_sum , window_sum = 0, 0
    window_start = 0
    for window_end in range(len(arr)):
        window_sum += arr[window_end]  # add the next element
        # slide the window, we don't need to slide if we've not hit the required window size of 'k'
        if window_end >= k-1:
            max_sum = max(max_sum, window_sum)
            window_sum -= arr[window_start]  # subtract the element going out
            window_start += 1  # slide the window ahead
    return max_sum

# Smallest subarray with given sum [SLIDING WINDOW PATTERN]
def smallest_subarray_with_given_sum(s):
    resp, tmp_sum = float('inf'), 0
    start = 0
    for i in range(len(arr)):
        tmp_sum += arr[i]
        while tmp_sum >= s:
            resp = min(resp, (i - start + 1))
            tmp_sum -= arr[start]
            start += 1
    return resp if resp != float('inf') else 0

# Longest substring with K distinct characters [SLIDING WINDOW PATTERN]
def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
    
    # Using simple hash map
    if k == 0:
        return 0
    res, dist_chars = 0, 0
    i,j = 0,0
    d = {}
    while j < len(s):
        if s[j] not in d:
            d[s[j]] = 1
        # If its a non-distinct char, just increment the counter
        else:
            d[s[j]] += 1
        j += 1

        # Clean way of doing the internal logic
        while len(d) > k:
            d[s[i]] -= 1
            if d[s[i]] == 0:
                del d[s[i]]
            i += 1
        res = max(res, j-i)
    return res


# https://leetcode.com/problems/minimum-window-substring/
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # Solution from EPI problem 12.6
        chars_to_cover = {}
        res = [1, float('inf')]
        for char in t:
            if char not in chars_to_cover:
                chars_to_cover[char] = 0
            chars_to_cover[char] += 1
        
        remaining_to_cover = len(t)
        left = 0
        for right, char in enumerate(s):
            if char in chars_to_cover:
                chars_to_cover[char] -= 1
                if chars_to_cover[char] >= 0:
                    remaining_to_cover -= 1
            
            while remaining_to_cover == 0:
                
                if (right - left) < (res[1] - res[0]):
                    res[0], res[1] = left, right
                if s[left] in chars_to_cover:
                    chars_to_cover[s[left]] += 1
                    if chars_to_cover[s[left]] > 0:
                        remaining_to_cover += 1

                left += 1
        
        return '' if res[1] == float('inf') else s[res[0]: res[1]+1]



def minDominoRotations(A, B):
    # Using sets
    import functools
    # Using set to find the min rotations
    s = functools.reduce(operator.and_, [set(d) for d in zip(A, B)])
    if not s: return -1
    x = s.pop()
    return min(len(A) - A.count(x), len(B) - B.count(x))



"""
Reading from files
"""
# Read chunks/blocks from a large file
def blocks(files, size=65536):
    while True:
        b = files.read(size)
        if not b: break
        yield b

with open("file", "r") as f:
    print (sum(bl.count("\n") for bl in blocks(f)))

# Reading files line by line (memory effecient)
with open('data.txt', 'r+b') as f:
    for line in f:
        print(line)


# https://leetcode.com/problems/longest-consecutive-sequence (Microsoft Onsite!)
def longestConsecutive(nums):
    longest_streak = 0
    num_set = set(nums)

    for num in num_set:
        # To find the first number in a sequence
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1
            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            longest_streak = max(longest_streak, current_streak)
    return longest_streak

# https://leetcode.com/problems/longest-increasing-subsequence (O(n2))
def longestIncSubsequence(nums):

    if not nums or len(nums) == 1:
        return nums
    n = len(nums)
    dp = [1] * n

    for i in range(1,n):
        for j in range(i):
            if nums[j] < nums[i] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1

    longest = 0
    for i in range(n):
        longest = max(longest, dp[i])
    return longest


# https://leetcode.com/problems/palindrome-pairs
def pal_pairs(words):
    wordict = {}
    res = []
    for i in range(len(words)):
        wordict[words[i]] = i
    for i in range(len(words)):
        for j in range(len(words[i]) + 1):
            tmp1 = words[i][:j]
            tmp2 = words[i][j:]
            if tmp1[::-1] in wordict and wordict[tmp1[::-1]] != i and tmp2 == tmp2[::-1]:
                res.append([i, wordict[tmp1[::-1]]])
            if j != 0 and tmp2[::-1] in wordict and wordict[tmp2[::-1]] != i and tmp1 == tmp1[::-1]:
                res.append([wordict[tmp2[::-1]], i])

    return res


# https://leetcode.com/discuss/interview-question/algorithms/125172/find-the-minimum-distance-between-two-numbers
def min_distance_between_two_elements(nums, x, y):
    cur_index = -1
    min_distance = float('inf')
    if len(nums) < 2:
        return 0
    for i in range(len(nums)):
        if nums[i] == x or nums[i] == y:
            if cur_index > -1 and nums[cur_index] != nums[i]:
                min_distance = min(min_distance, i-cur_index)
            cur_index = i
    return min_distance


def delete_dup_in_sorted_list(nums):
    if len(nums) < 2:
        return nums
    replace_pointer, cur = 0, 1
    while cur < len(nums):
        if nums[cur] == nums[replace_pointer]:
            pass
        else:
            replace_pointer += 1
            nums[replace_pointer] = nums[cur]
        cur += 1

    return nums[:replace_pointer+1]

# https://leetcode.com/problems/sort-colors/
# Low will always point to the first one
# Equal will point to next of recent 1
# High will point to location, that is to be swapped with next high found with equal
def dutch_national_flag(nums, pivot=1):
    lo, equal, hi = 0, 0, len(nums)-1
    while equal < hi:
        if nums[equal] < pivot:
            nums[lo], nums[equal] = nums[equal], nums[lo]
            lo += 1
            equal += 1
        elif nums[equal] > pivot:
            nums[hi], nums[equal] = nums[equal], nums[hi]
            hi -= 1
        else:
            equal += 1
    return nums

def wiggle_sort(nums):
    for i in range(len(nums) - 1):
        if not i % 2 and nums[i] > nums[i + 1]:
            nums[i], nums[i + 1] = nums[i + 1], nums[i]
        if i % 2 and nums[i + 1] > nums[i]:
            nums[i], nums[i + 1] = nums[i + 1], nums[i]
    return nums


# https://leetcode.com/problems/sentence-similarity
class Solution(object):
    def areSentencesSimilar(self, words1, words2, pairs):
        from collections import defaultdict
        if len(words1) != len(words2):
            return False
        words = defaultdict(set) # Important data structure for this problem
        for word1, word2 in pairs:
            words[word1].add(word2)
            words[word2].add(word1)
        for word1, word2 in zip(words1, words2):
            if word1 != word2 and word2 not in words[word1]:
                return False
        return True


# (Similar to https://www.geeksforgeeks.org/maximum-sum-such-that-no-two-elements-are-adjacent)
# You are a treasure hunter planning to physically search for treasures  in the buildings along a street.
# Each building has a certain amount of money stashed, the only constraint stopping you from finding each of them
# is that adjacent buildings have dungeons and it will trap you if two adjacent buildings were searched into on
# the same night. Given an integer array nums representing the amount of money of each building, return the maximum
# amount of money you can find tonight without getting trapped in the dungeon.
# 
# Input: nums = [1,2,3,1] => 4
# nums = [5, 3, 4, 11, 2] => 16

def getMaxAmt(nums):
    if len(nums) == 0:
        return 0
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])
    maxSum = 0
    # this is a simple form (1D) of DP approach
    prevPrev, prev = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        maxSum = max(nums[i] + prevPrev, prev)
        prevPrev = prev
        prev = maxSum
    return maxSum


# https://leetcode.com/problems/coin-change
class Solution:
    def coinChangeBFS(self, coins: List[int], amount: int) -> int:
        q = deque([(amount, 0)])
        seen = set([amount]) # memoization/caching here
        while q:
            remainingAmount, num_coins = q.popleft()
            if remainingAmount == 0:
                    return num_coins
            for coin in coins:
                subAmount = remainingAmount - coin
                if subAmount >= 0 and subAmount not in seen:
                    q.append((subAmount, num_coins + 1))
                    seen.add(subAmount)
        return -1

    def coinChangeDFS(self, coins: List[int], amount: int) -> int:

        @lru_cache(None)
        def dfs(rem):
            if rem < 0:
                return -1
            if rem == 0:
                return 0
            min_cost = float('inf')
            for coin in coins:
                res = dfs(rem - coin)
                if res != -1:
                    min_cost = min(min_cost, res + 1)
            return min_cost if min_cost != float('inf') else -1

        return dfs(amount)
    

# https://leetcode.com/problems/longest-substring-without-repeating-characters/description/
def longest_nonrepeating_substring(input):
    res = 0
    helper = set()
    start, end = 0, 0
    while end < len(input):
        if input[end] not in helper:
            helper.add(input[end])
            res = max(res, end-start+1) # validate this
            end += 1
        else:
            helper.remove(input[start])
            start += 1
    return res
