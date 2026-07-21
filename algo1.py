# Two pointer
# Two Sum  - Input Array Is Sorted (Two Pointers)
# def twopinter(num: list[int], tar: int):
#     left = 0
#     right = len(num) - 1

#     while(left < right):
#         if num[left] + num[right] == tar:
#             return [left+1, right+1]
#         elif num[left] + num[right] < tar:
#             left += 1
#         else:
#             right -= 1

#     return []

# numbers = [2,4,3,5,7]
# print(twopinter(numbers, 7))  

# Problem 2: Container With Most Water (Two Pointers)

# def maxarea(height: list[int]):
#     left = 0
#     right = len(height) -1
#     max_water= 0

#     while left < right:
#         width = right - left
#         h = min(height[left], height[right])
#         max_water = max(max_water, width * h)

#         # Move the pointer with the shorter bar
#         if height[left] < height[right]:
#             left += 1
#         else:
#             right -= 1

#     return max_water

# arr = [2, 1, 8, 6, 4, 6, 5, 5]
# print(maxarea(arr))

# Problem 3: Maximum Average Subarray I (Fixed-Size Sliding Window)
# def findMaxAverage(nums: list[int], k):
#     current_sum = sum(nums[:k])
#     max_sum = current_sum

#     for i in range(k, len(nums)):
#         current_sum += nums[i] - nums[i-k]
#         max_sum = max(max_sum, current_sum)
    
#     return max_sum/k;

# arr = [2, 1, 8, 6, 4, 6, 5, 5]
# print(findMaxAverage(arr, 3))

# Problem 4: Longest Substring Without Repeating Characters (Variable Sliding Window)

# s = "consistency"

# def lengthOfLongestSubstring(s):
#     seen = set()
#     left = 0
#     max_len = 0

#     for right in range(len(s)):
#         while s[right] in seen:
#             seen.remove(s[left])
#             left += 1

#         seen.add(s[right])
#         max_len = max(max_len, right - left + 1)

#     return max_len

# print(lengthOfLongestSubstring(s))

