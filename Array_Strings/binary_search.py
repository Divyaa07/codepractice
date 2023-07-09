# Tips:
# While reducing search space, most probably you will need to return the left pointer
# 

# https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
def min_sorted_rotated(arr):
    # Recursive
    def find_min_elem(arr, start, end):
        if start == end:
            return arr[start]
        m = start + (end-start)//2

        if arr[m] < arr[m-1]:
            return arr[m]
        elif arr[m] > arr[end]:
            find_min_elem(arr, m+1, end)
        # if arr[start] < arr[m]:
            # find_min_elem(arr, m+1, end)
        else:
            find_min_elem(arr, start, m)
    return find_min_elem(arr, 0, len(arr)-1)

# Only repeating element in a sorted array of size n, elements from 1...n
def sorted_repeating_element():
    pass

# Single element in a sorted array with every element twice except one
# https://leetcode.com/problems/single-element-in-a-sorted-array/
def single_non_dup_element_2(nums):
    # Method 1: Using binary search (Reducing search space)
    # If you are returning from inside the loop, use left <= right
    # If you are reducing the search space, use left < right and finally return a[left]
    l,r = 0, len(nums)-1
    mid = 0
    while l < r:
        mid = l + (r-l)//2
        if nums[mid] == nums[mid^1]:
            # Search in right side
            l = mid+1
        else:
            # Search in left half. Since mid not equal, it might as well be the answer, so include it!
            r = mid
    return nums[l]
    
    # Method 2: Using XOR operator
    res = nums[0]
    for i in range(1, len(nums)):
        res = res ^ nums[i]
    return res

# https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
"""
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
"""
def searchRange(self, nums: List[int], target: int) -> List[int]:
    def first_pos():
        result = -1
        left, right = 0, len(nums)-1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                result = mid
                right = mid - 1
        return result
    
    def last_pos():
        result = -1
        left, right = 0, len(nums)-1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                result = mid
                left = mid + 1
        return result

    return (first_pos(), last_pos())


# https://leetcode.com/problems/search-insert-position/
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
                
        left, right = 0, len(nums)-1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                return mid
        
        return left


# MORE BINARY SEARCH PROBLEMS
# https://leetcode.com/problems/koko-eating-bananas/
# https://leetcode.com/problems/minimized-maximum-of-products-distributed-to-any-store/
# https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/
# https://leetcode.com/problems/minimize-max-distance-to-gas-station/
# https://leetcode.com/problems/split-array-largest-sum/
# https://leetcode.com/problems/divide-chocolate/
# https://leetcode.com/problems/kth-missing-positive-number/
# https://leetcode.com/problems/maximum-value-at-a-given-index-in-a-bounded-array/
# https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/
# https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/
