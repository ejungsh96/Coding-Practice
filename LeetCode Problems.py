704. Binary Search
Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.
You must write an algorithm with O(log n) runtime complexity.


Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4


1. [-1,0,3,5,9,12]
 beg=0        end= len(nums) - 1
       mid = (beg + end) // 2
2. if nums[mid] == target return mid
3. if nums[mid] < target then beg += 1
else then end -= 1
4. return -1




class Solution:
    def search(self, nums: List[int], target: int) -> int:
        beginning, end = 0, len(nums) - 1
        
        while beginning <= end:
            mid = (beginning + end) // 2
            
            if nums[mid] == target:
                return mid
            
            if nums[mid] < target:
                beginning = mid + 1
            else:
                 end = mid - 1
        
        return -1


















278. First Bad Version
You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.
Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.
You are given an API bool isBadVersion(version) which returns whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.
   1. [1, 2, 3, ... , n]
beg  mid    end
0     n/2     n
   2. [F, F, F, F, F, F, F, T, T, T, T, T]
beg             I                 end
   3. mid = (beg + end) // 2
   4. while


# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:


class Solution:
    def firstBadVersion(self, n: int) -> int:
        beg, end = 1, n
        
        while beg < end:
            mid = beg + (end - beg) // 2
            if isBadVersion(mid) == True:
                end = mid
            if isBadVersion(mid) == False:
                beg = mid + 1
        
        return beg






35. Search Insert Position
Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
You must write an algorithm with O(log n) runtime complexity.
Example 1:
Input: nums = [1,3,5,6], target = 5
Output: 2


      1. left, right = 0, len(nums) - 1
      2. while left < right:
   mid = left + (right - left) // 2 
   if nums[mid] == target return mid
   if nums[mid] > target:
       right = mid - 1
   else:
       left = mid + 1


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        if target < nums[0]:
            return 0
        if target > nums[-1]:
            return len(nums)
        if target == nums[0]:
            return 0
        
        while left < right:
            mid = left + (right - left) // 2 
            
            if nums[mid] == target:
                return mid
            if right - left == 1:
                return right
            if nums[mid] > target:
                right = mid
            else:
                left = mid


977. Squares of a Sorted Array
Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.
Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].
         1. [-4,-1,0,3,10] -> 16, 1, 0, 9, 100 -> sort : O(nlogn), O(n)
         2. two pointer method
         3. L          R
         4. Use collections.deque to append left side of the list
         5. if abs(L) > abs(R) then output.appendleft(nums[L]**2), left += 1
         6. else then output.appendleft(nums[R]**2), right -= 1
         7. return output


from collections import deque
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        output = deque()
        left, right = 0, len(nums) - 1
        while left <= right:
            if abs(nums[left]) > abs(nums[right]):
                output.appendleft(nums[left]**2)
                left += 1
            else:
                output.appendleft(nums[right]**2)
                right -= 1
        
        return list(output)






































189. Rotate Array


Given an array, rotate the array to the right by k steps, where k is non-negative.


Example 1:
Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]
         1. left, right = 0, len(nums) - 1
         2. iterate k times
   for n
    tmp = nums[left]
    nums[left] = nums[right]




    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)


        # Solution 1: Brute Force
        # end = len(nums) - 1
        # for _ in range(k):
        #     tmp = nums[end]
        #     for i in range(len(nums)):
        #         nums[i], tmp = tmp, nums[i]
        
        # Solution 1: trick 
        n = len(nums)
        nums[:] = nums[n-k:] + nums[:n-k]
















283. Move Zeroes


Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.
Note that you must do this in-place without making a copy of the array.
Example 1:
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]


            1. count = 0
for i in nums
 if i == 0 then nums.remove(0), count += 1
            2. for i in range(count)
 nums.append(0)
            3. return nums




class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        slow = 0
        
        for fast in range(len(nums)):
            if nums[fast] != 0 and nums[slow] == 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                
            if nums[slow] != 0:
                slow += 1
















167. Two Sum II - Input Array Is Sorted


Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.
Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.
The tests are generated such that there is exactly one solution. You may not use the same element twice.
Your solution must use only constant extra space.


Example 1:
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].


               1. for i in numbers find target - i in the rest of the list O(n^2)




class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1
        
        while left <= right:
            s = numbers[left] + numbers[right]
            
            if s == target:
                return [left + 1, right + 1]
            elif s < target:
                left += 1
            else:
                right -= 1
344. Reverse String


Write a function that reverses a string. The input string is given as an array of characters s.
You must do this by modifying the input array in-place with O(1) extra memory.
Example 1:
Input: s = ["h","e","l","l","o"]
Output: ["o","l","l","e","h"]
First solution: return s.reverse()
Second solution: Two pointers
               1. left, right as the first index and the last index
               2. while left <= right
               3. swap left and right
               4. increment left, decrement right by 1


left, right = 0, len(s) - 1
while left <= right:
  s[left], s[right] = s[right], s[left]
  left += 1
  right -= 1




class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        left, right = 0, len(s) -1
        
        while left <= right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1














557. Reverse Words in a String III


Given a string s, reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.


Example 1:
Input: s = "Let's take LeetCode contest"
Output: "s'teL ekat edoCteeL tsetnoc"


               1. split it -> sentence = s.split(“ ”)
               2. reverse each word and contatonate it to “”
               3. return [:-1]


class Solution:
    def reverseWords(self, s: str) -> str:
        sentence = s.split(" ")
        output = ""
        for word in sentence:
            output = output + word[::-1] + " "
        
        return output[:-1]
















































876. Middle of the Linked List


Given the head of a singly linked list, return the middle node of the linked list.
If there are two middle nodes, return the second middle node.
Example 1:
  

Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.
               1. travel through from the first node to the end -> count the number of nodes
               2. calculate the number of middle index and travel from the first  node again
               3. T: O(n) S: O(n)


               1. slow and fast pointer: when slow traverses 1, fast traverses 2
               2. when fast is the end of the linked list, slow will be in the middle


class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # Output to List T: O(n), S: O(n)
        lst = []
        ptr = head
        while ptr != None:
            lst.append(ptr.val)
            ptr = ptr.next
        index = len(lst) // 2 + 1
        
        for i in range(index - 1):
            head = head.next
        return head


        # Slow and Fast -> T: O(n), S: O(1)
        slow = fast = head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
        return slow
19. Remove Nth Node From End of List


Given the head of a linked list, remove the nth node from the end of the list and return its head.
Example 1:
  

Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
               1. we can count the number of nodes, subtract the nth node -> index of the node
               2. traverse from the first


               1. slow, fast -> nth node difference between them
               2. when fast is the end node of the list -> slow is what we’re going to remove
               3. remove -> slow.next = slow.next.next if slow.next.next exist, or None


class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        slow = fast = head
        
        for _ in range(n):
            fast = fast.next
        if fast == None:
            return head.next
        
        while fast.next != None:
            fast = fast.next
            slow = slow.next
            
        slow.next = slow.next.next


        
        return head












3. Longest Substring Without Repeating Characters


Given a string s, find the length of the longest substring without repeating characters.


Example 1:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
               1. set to check if it is a unique char.
               2. left, right
               3. right traverses until the end
if s[right] is in the set, then remove(s[left])






class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        seen_char = set()
        left = 0
        max_count = 0
        for right in range(len(s)):
            while s[right] in seen_char:
                seen_char.remove(s[left])
                left += 1
            seen_char.add(s[right])
            max_count = max(max_count, right - left + 1)
        
        return max_count


































567. Permutation in String#


Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.
In other words, return true if one of s1's permutations is the substring of s2.
Example 1:
Input: s1 = "ab", s2 = "eidbaooo"
Output: true
Explanation: s2 contains one permutation of s1 ("ba").
                  1. create a set and put s2 in it.
                  2. check if all of character of s1 is in the set


class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        counter, l = Counter(s1), len(s1)
        
        for i in range(len(s2)):
            if s2[i] in counter:
                counter[s2[i]] -= 1
            if i >= l and s2[i - l] in counter:
                counter[s2[i - l]] += 1
            if all([counter[i] == 0 for i in counter]):
                return True
        
        return False


733. Flood Fill#
An image is represented by an m x n integer grid image where image[i][j] represents the pixel value of the image.
You are also given three integers sr, sc, and newColor. You should perform a flood fill on the image starting from the pixel image[sr][sc].
To perform a flood fill, consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color), and so on. Replace the color of all of the aforementioned pixels with newColor.
Return the modified image after performing the flood fill.
Example 1:
  

Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2
Output: [[2,2,2],[2,2,0],[2,0,1]]
Explanation: From the center of the image with position (sr, sc) = (1, 1) (i.e., the red pixel), all pixels connected by a path of the same color as the starting pixel (i.e., the blue pixels) are coloured with the new color.
Note the bottom corner is not colored 2, because it is not 4-directionally connected to the starting pixel.
                  1. DFS -> Recursion




class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        l_row, l_colum = len(image), len(image[0])
        color = image[sr][sc]
        if color == newColor:
            return image
        
        def dfs(r, c):
            if image[r][c] == color:
                image[r][c] = newColor
                if r >= 1:
                    dfs(r - 1, c)
                if r + 1 < l_row:
                    dfs(r + 1, c)
                if c >= 1:
                    dfs(r, c - 1)
                if c + 1 < l_colum:
                    dfs(r, c + 1)
        
        dfs(sr, sc)
        return image






695. Max Area of Island
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.
The area of an island is the number of cells with a value 1 in the island.
Return the maximum area of an island in grid. If there is no island, return 0.
Example 1:
  

Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
Output: 6
Explanation: The answer is not 11, because the island must be connected 4-directionally.
                  1. return the max_area
                  2. recursively add up the area of the island -> DFS
                  3. if grid[row][col] == 1






class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        ROWS, COLS = len(grid), len(grid[0])
        visit = set()
        
        def dfs(r, c):
            if (r < 0 or r == ROWS or c < 0 or c == COLS or grid[r][c] == 0 or (r, c) in visit):
                return 0
            visit.add((r, c))
            return (1 + dfs(r + 1, c) +
                        dfs(r - 1, c) +
                        dfs(r, c + 1) +
                        dfs(r, c - 1))
        
        area = 0
        for r in range(ROWS):
            for c in range(COLS):
                area = max(area, dfs(r, c))
        return area










617. Merge Two Binary Trees
You are given two binary trees root1 and root2.
Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. You need to merge the two trees into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of the new tree.
Return the merged tree.
Note: The merging process must start from the root nodes of both trees.
Example 1:
  

Input: root1 = [1,3,2,5], root2 = [2,1,3,null,4,null,7]
Output: [3,4,5,5,4,null,7]
                  1. DFS -> ptr1 at root1, ptr2 at root2
                  2. traverses at the same time
                  3. add the value and create the return tree
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root1 and not root2:
            return None
        
        value1 = root1.val if root1 else 0
        value2 = root2.val if root2 else 0
        root = TreeNode(value1 + value2)
        
        root.left = self.mergeTrees(root1.left if root1 else None, root2.left if root2 else None)
        root.right = self.mergeTrees(root1.right if root1 else None, root2.right if root2 else None)
        
        return root
































116. Populating Next Right Pointers in Each Node
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}


Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.
Initially, all next pointers are set to NULL.
Example 1:
  

Input: root = [1,2,3,4,5,6,7]
Output: [1,#,2,3,#,4,5,6,7,#]
Explanation: Given the above perfect binary tree (Figure A), your function should populate each next pointer to point to its next right node, just like in Figure B. The serialized output is in level order as connected by the next pointers, with '#' signifying the end of each level.
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return None
        if not root.left and not root.right:
            return root
        
        root.left.next = root.right
        if root.next:
            root.right.next = root.next.left
            
        self.connect(root.left)
        self.connect(root.right)
        return root




















542. 01 Matrix#
Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.
The distance between two adjacent cells is 1.
Example 1:
  

Input: mat = [[0,0,0],[0,1,0],[0,0,0]]
Output: [[0,0,0],[0,1,0],[0,0,0]]


BFS -> Queue


class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        ROWS, COLS = len(mat), len(mat[0])
        DIR = [0, 1, 0, -1, 0]
        
        q = deque([])
        for r in range(ROWS):
            for c in range(COLS):
                if mat[r][c] == 0:
                    q.append((r, c))
                else:
                    mat[r][c] = -1
        
        while q:
            r, c = q.popleft()
            for i in range(4):
                nr, nc = r + DIR[i], c + DIR[i + 1]
                if nr < 0 or nr == ROWS or nc < 0 or nc == COLS or mat[nr][nc] != -1:
                    continue
                mat[nr][nc] = mat[r][c] + 1
                q.append((nr, nc))
        return mat






















994. Rotting Oranges
You are given an m x n grid where each cell can have one of three values:
                  * 0 representing an empty cell,
                  * 1 representing a fresh orange, or
                  * 2 representing a rotten orange.
Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.
Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.
Example 1:
  

Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        q = deque()
        time, fresh = 0, 0
        
        ROWS, COLS = len(grid), len(grid[0])
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r][c] == 1:
                    fresh += 1
                if grid[r][c] == 2:
                    q.append((r, c))
                    
        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        while q and fresh > 0:
            for i in range(len(q)):
                r, c = q.popleft()
                for dr, dc in directions:
                    row, col = dr + r, dc + c
                    # if in bounds and fresh, make rotten
                    if (row < 0 or row == ROWS or col < 0 or col == COLS or grid[row][col] != 1):
                        continue
                    grid[row][col] = 2
                    q.append((row, col))
                    fresh -= 1
            time += 1
        
        return time if fresh == 0 else -1










21. Merge Two Sorted Lists
You are given the heads of two sorted linked lists list1 and list2.
Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.
Return the head of the merged linked list.
Example 1:
  

Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
                  1. Iterative solution:
1) Make a tmp linked list to have a list to insert nodes. Insert tail
2) iterate while list1 and list2 are not None
if list1.val < list2.val then tail.next = list1
list1 = list1.next
else tail.next = list2
list2 = list2.next
tail = tail.next
3) And if one of the list is now exhausted, append the rest of it at tail
if list1 is not None then tail.next = list1
elif list 2 is not None then tail.next = list2
4) return tmp.next (as tmp was only for returning the output)
                  2. Recursive solution:
1) Base Case: if not list1 then return list1
             if not list2 then return list2
2) if list1.val < list2.val:
      list1.next = self.mergeTwoList(list1.next, list2)
      return list1
3) else:
      list2.next = self.mergeTwoList(list1, list2.next)
      return list2








































206. Reverse Linked List
Given the head of a singly linked list, reverse the list, and return the reversed list.
Example 1:
  

Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
                     1. make a tmp node and tmp.next = head
                     2. left, right = 


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, cur = None, head
        while cur:
            cur.next, prev, cur = prev, cur, cur.next
        
        return prev




122. Best Time to Buy and Sell Stock II
You are given an integer array prices where prices[i] is the price of a given stock on the ith day.
On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.
Find and return the maximum profit you can achieve.
Example 1:
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.
                     1. track the lowest, highest prices
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                profit += (prices[i] - prices[i - 1])
        
        return profit


If the problem is about the maximum value according to time -> draw a graph!
217. Contains Duplicate
Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
Example 1:
Input: nums = [1,2,3,1]
Output: true
                     1. Use the fact that a set doen’t allow any duplication.


class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return False if len(nums) == len(set(nums)) else True


























136. Single Number
Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.
You must implement a solution with a linear runtime complexity and use only constant extra space.
Example 1:
Input: nums = [2,2,1]
Output: 1


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        counter = Counter(nums)
        
        for i in counter:
            if counter[i] == 1:
                return i
















350. Intersection of Two Arrays II
Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays and you may return the result in any order.
Example 1:
Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]


































941. Valid Mountain Array
Given an array of integers arr, return true if and only if it is a valid mountain array.
Recall that arr is a mountain array if and only if:
                     * arr.length >= 3
                     * There exists some i with 0 < i < arr.length - 1 such that:
                     * arr[0] < arr[1] < ... < arr[i - 1] < arr[i]
                     * arr[i] > arr[i + 1] > ... > arr[arr.length - 1]
  

 
Example 1:
Input: arr = [2,1]
Output: false


class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        if len(arr) < 3:
            return False
        
        # compare to the next element, if it is decreaes or same (plateou) return False
        # Boolean to check
        
        left, right = 0, len(arr) - 1
        
        while (arr[left] < arr[left + 1]) and (left + 1 < len(arr) - 1):
            left += 1
        
        while (arr[right] < arr[right - 1]) and (right - 1 > 0):
            right -= 1
        
        return left == right






















939. Minimum Area Rectangle
You are given an array of points in the X-Y plane points where points[i] = [xi, yi].
Return the minimum area of a rectangle formed from these points, with sides parallel to the X and Y axes. If there is not any such rectangle, return 0.
 
Example 1:
  

Input: points = [[1,1],[1,3],[3,1],[3,3],[2,2]]
Output: 4


class Solution:
    def minAreaRect(self, points: List[List[int]]) -> int:
        # 1. find all available rectangles areas
        # return the minimum one
        
        seen = set()
        res = float('inf')
        for x1, y1 in points:
            for x2, y2 in seen:
                if (x1, y2) in seen and (x2, y1) in seen:
                    area = abs(x1 - x2) * abs(y1 - y2)
                    if area and area < res:
                        res = area
            seen.add((x1, y1))
        return res if res < float('inf') else 0


































731. My Calendar II
You are implementing a program to use as your calendar. We can add a new event if adding the event will not cause a triple booking.
A triple booking happens when three events have some non-empty intersection (i.e., some moment is common to all the three events.).
The event can be represented as a pair of integers start and end that represents a booking on the half-open interval [start, end), the range of real numbers x such that start <= x < end.
Implement the MyCalendarTwo class:
                     * MyCalendarTwo() Initializes the calendar object.
                     * boolean book(int start, int end) Returns true if the event can be added to the calendar successfully without causing a triple booking. Otherwise, return false and do not add the event to the calendar.
 
Example 1:
Input
["MyCalendarTwo", "book", "book", "book", "book", "book", "book"]
[[], [10, 20], [50, 60], [10, 40], [5, 15], [5, 10], [25, 55]]
Output
[null, true, true, true, false, true, true]


Explanation
MyCalendarTwo myCalendarTwo = new MyCalendarTwo();
myCalendarTwo.book(10, 20); // return True, The event can be booked. 
myCalendarTwo.book(50, 60); // return True, The event can be booked. 
myCalendarTwo.book(10, 40); // return True, The event can be double booked. 
myCalendarTwo.book(5, 15);  // return False, The event cannot be booked, because it would result in a triple booking.
myCalendarTwo.book(5, 10); // return True, The event can be booked, as it does not use time 10 which is already double booked.
myCalendarTwo.book(25, 55); // return True, The event can be booked, as the time in [25, 40) will be double booked with the third event, the time [40, 50) will be single booked, and the time [50, 55) will be double booked with the second event.


class MyCalendarTwo:


    def __init__(self):
        self.overlaps = []
        self.calendar = []


    def book(self, start, end):
        for i, j in self.overlaps:
            if start < j and end > i:
                return False
        for i, j in self.calendar:
            if start < j and end > i:
                self.overlaps.append((max(start, i), min(end, j)))
        self.calendar.append((start, end))
        return True










































350. Intersection of Two Arrays II
Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays and you may return the result in any order.
 
Example 1:
Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]


class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # Brute Force -> O(n^2)
        output = []
        for i in nums1:
            if i in nums2:
                output.append(i)
                nums2.remove(i)
        
        return output
        
        # Counter -> O(n^2)
        counter1, counter2 = Counter(nums1), Counter(nums2) # O(n)
        
        output = []
        for key, value in counter1.items(): # O(n)
            if key in counter2:
                m = min(value, counter2[key])   # O(n)
                for _ in range(m):  # O(n)
                    output.append(key)
        
        return output
        
        # Sort and Two Pointer
        nums1.sort()    # O(nlogn)
        nums2.sort()
        
        pointer1 = pointer2 = 0
        
        output = []
        while pointer1 < len(nums1) and pointer2 < len(nums2):
            if nums1[pointer1] == nums2[pointer2]:
                output.append(nums1[pointer1])
                pointer1 += 1
                pointer2 += 1
            elif nums1[pointer1] < nums2[pointer2]:
                pointer1 += 1
            else:
                pointer2 += 1
        
        return output


        # Solution: Convert nums1 to a dictionary and iterate nums2 according to the dict
      




































2022. Convert 1D Array Into 2D Array
You are given a 0-indexed 1-dimensional (1D) integer array original, and two integers, m and n. You are tasked with creating a 2-dimensional (2D) array with m rows and n columns using all the elements from original.
The elements from indices 0 to n - 1 (inclusive) of original should form the first row of the constructed 2D array, the elements from indices n to 2 * n - 1 (inclusive) should form the second row of the constructed 2D array, and so on.
Return an m x n 2D array constructed according to the above procedure, or an empty 2D array if it is impossible.
Example 1:
  

Input: original = [1,2,3,4], m = 2, n = 2
Output: [[1,2],[3,4]]
Explanation: The constructed 2D array should contain 2 rows and 2 columns.
The first group of n=2 elements in original, [1,2], becomes the first row in the constructed 2D array.
The second group of n=2 elements in original, [3,4], becomes the second row in the constructed 2D array.
class Solution:
    def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
        if len(original) != m * n:
            return []
        
        output = []
        for _ in range(m):
            output.append(original[:n])
            original = original[n:]
        
        return output


























79. Word Search
Given an m x n grid of characters board and a string word, return true if word exists in the grid.
The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.
Example 1:
  

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        ROWS, COLS = len(board), len(board[0])
        seen = set()
        
        def dfs(row, col, word, compare):
            if row < 0 or row == ROWS or col < 0 or col == COLS or (row, col) in seen:
                return "", False
            compare += board[row][col]
            seen.add((row, col))
            print(compare)
            
            if compare == word:
                return compare, True
            else:
                directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    dfs(nr, nc, word, compare)
        
        output = False
        for r in range(ROWS):
            for c in range(COLS):
                output = output or dfs(r, c, word, "")
        
        return output






66. Plus One
You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.
Increment the large integer by one and return the resulting array of digits.
Example 1:
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
Incrementing by one gives 123 + 1 = 124.
Thus, the result should be [1,2,4].


class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        to_string = ""
        for i in digits:
            to_string += str(i)
        
        output = int(to_string) + 1
        
        return str(output)






1. Two Sum
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.
Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = dict()
        
        for i in range(len(nums)):
            comp = target - nums[i]
            
            if nums[i] in seen:
                return [seen[nums[i]], i]
            else:
                seen[comp] = i




36. Valid Sudoku
Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
                     1. Each row must contain the digits 1-9 without repetition.
                     2. Each column must contain the digits 1-9 without repetition.
                     3. Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
Note:
                     * A Sudoku board (partially filled) could be valid but is not necessarily solvable.
                     * Only the filled cells need to be validated according to the mentioned rules.
Example 1:
  

Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        cols = collections.defaultdict(set)
        rows = collections.defaultdict(set)
        squares = collections.defaultdict(set) # key = (r/3, c/3)
        
        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                if (board[r][c] in rows[r] or 
                   board[r][c] in cols[c] or
                   board[r][c] in squares[(r // 3, c // 3)]):
                    return False
                cols[c].add(board[r][c])
                rows[r].add(board[r][c])
                squares[(r // 3, c // 3)].add(board[r][c])
        
        return True




48. Rotate Image
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.
Example 1:
  

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        l, r = 0, len(matrix) - 1
        
        while l < r:
            # a layer
            for i in range(r - l):
                top, bottom = l, r
                
                # save the topleft
                top_left = matrix[top][l + i]
                
                # move bottom left into top left
                matrix[top][l + i] = matrix[bottom - i][l]
                
                # move bottom right into bottom left
                matrix[bottom - i][l] = matrix[bottom][r - i]
                
                # move top right into bottom right
                matrix[bottom][r - i] = matrix[top + i][r]
                
                # move top left into top right
                matrix[top + i][r] = top_left
            
            # shift to inner layer
            r -= 1
            l += 1
            
387. First Unique Character in a String
Given a string s, find the first non-repeating character in it and return its index. If it does not exist, return -1.
Example 1:
Input: s = "leetcode"
Output: 0
 
class Solution:
    def firstUniqChar(self, s: str) -> int:
        # Counter O(n^2)
        counter = Counter(s) # O(n)
        
        for index, char in enumerate(s): # O(n)
            if counter[char] == 1:
                return index # O((n-m)*m) where n: len(string), m: len(substring)
        
        return -1
 
 
 
 
 
 
 
242. Valid Anagram
Given two strings s and t, return true if t is an anagram of s, and false otherwise.
An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.
Example 1:
Input: s = "anagram", t = "nagaram"
Output: true
 
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # Sorting
        return sorted(s) == sorted(t)
    
        # Hash Table
        if len(s) != len(t): return False 
    
        counter = {}
        for i in s:
            if i in counter:
                counter[i] += 1
            else:
                counter[i] = 1
        
        for j in t:
            if j not in counter:
                return False
            else:
                counter[j] -= 1
        
        for value in counter.values():
            if value != 0:
                return False
        
        return True
 
 
 
 
 
 
 
 
 
 
 
 
 
 
125. Valid Palindrome
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.
Given a string s, return true if it is a palindrome, or false otherwise.
Example 1:
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
class Solution:
    def isPalindrome(self, s: str) -> bool:
        converted = [i for i in s.lower() if i.isalnum()]
        
        # reverse
        return converted == converted[::-1]
    
        # two pointer    
        left, right = 0, len(converted) - 1
        
        while left < right:
            if converted[left] != converted[right]:
                return False
            left += 1
            right -= 1
        
        return True


# Meta Coding Interview Prep 1
def rotationalCipher(input, rotation_factor):
  # Write your code here
  output = ""
  for i in range(len(input)):
    
    if input[i].isalnum():
      if input[i].isalpha():
        if input[i].isupper():
          output += chr((ord(input[i]) + rotation_factor - 65) % 26 + 65)
        else:
          output += chr((ord(input[i]) + rotation_factor - 97) % 26 + 97)
      if input[i].isnumeric():
        # print("number ", int(input[i]) + rotation_factor)
        if int(input[i]) + rotation_factor < 10:
          output += str(int(input[i]) + rotation_factor)
          # print("1: ", str(int(input[i]) + rotation_factor))
        else:
          output += str((int(input[i]) + rotation_factor) % 10)
          # print("2: ", str(int(input[i]) + rotation_factor % 10))
    else:
      output = output + input[i]
  return output

162. Find Peak Element
A peak element is an element that is strictly greater than its neighbors.
Given an integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.
You may imagine that nums[-1] = nums[n] = -∞.
You must write an algorithm that runs in O(log n) time.
Example 1:
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
 
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        # Linear Search
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                return i
        
        return len(nums) - 1
    
        # Recursive Binary Search
        def search(nums, left, right):
            if left == right:
                return left
            mid = (left + right) // 2
            if nums[mid] > nums[mid + 1]:
                return search(nums, left, mid)
            return search(nums, mid + 1, right)
        
        return search(nums, 0, len(nums) - 1)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
394. Decode String
Given an encoded string, return its decoded string.
The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.
You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc.
Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there will not be input like 3a or 2[4].
Example 1:
Input: s = "3[a]2[bc]"
Output: "aaabcbc"
 
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        
        for i in range(len(s)):
            # put into the stack until it is "]"
            if s[i] != "]":
                stack.append(s[i])
            else:
                # parse the substring
                substr = ""
                while stack[-1] != "[":
                    substr = stack.pop() + substr
                stack.pop()
                
                # parse the number
                k = ""
                while stack and stack[-1].isdigit():
                    k = stack.pop() + k
                
                # put the multiplied string back into the stack
                stack.append(int(k) * substr)
        
        # all are decoded, return in a string form
        return "".join(stack)
 
 
 
 
 
 
 
 
 
 
 
8. String to Integer (atoi)
Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer (similar to C/C++'s atoi function).
The algorithm for myAtoi(string s) is as follows:
Read in and ignore any leading whitespace.
Check if the next character (if not already at the end of the string) is '-' or '+'. Read this character in if it is either. This determines if the final result is negative or positive respectively. Assume the result is positive if neither is present.
Read in next the characters until the next non-digit character or the end of the input is reached. The rest of the string is ignored.
Convert these digits into an integer (i.e. "123" -> 123, "0032" -> 32). If no digits were read, then the integer is 0. Change the sign as necessary (from step 2).
If the integer is out of the 32-bit signed integer range [-231, 231 - 1], then clamp the integer so that it remains in the range. Specifically, integers less than -231 should be clamped to -231, and integers greater than 231 - 1 should be clamped to 231 - 1.
Return the integer as the final result.
Note:
Only the space character ' ' is considered a whitespace character.
Do not ignore any characters other than the leading whitespace or the rest of the string after the digits.
Example 1:
Input: s = "42"
Output: 42
Explanation: The underlined characters are what is read in, the caret is the current reader position.
Step 1: "42" (no characters read because there is no leading whitespace)
         ^
Step 2: "42" (no characters read because there is neither a '-' nor '+')
         ^
Step 3: "42" ("42" is read in)
           ^
The parsed integer is 42.
Since 42 is in the range [-231, 231 - 1], the final result is 42.
 
class Solution:
    def myAtoi(self, s: str) -> int:
        sign = 1 
        result = 0
        index = 0
        n = len(s)
        
        INT_MAX = pow(2,31) - 1 
        INT_MIN = -pow(2,31)
        
        # Discard all spaces from the beginning of the input string.
        while index < n and s[index] == ' ':
            index += 1
        
        # sign = +1, if it's positive number, otherwise sign = -1. 
        if index < n and s[index] == '+':
            sign = 1
            index += 1
        elif index < n and s[index] == '-':
            sign = -1
            index += 1
        
        # Traverse next digits of input and stop if it is not a digit. 
        # End of string is also non-digit character.
        while index < n and s[index].isdigit():
            digit = int(s[index])
            
            # Check overflow and underflow conditions. 
            if ((result > INT_MAX // 10) or (result == INT_MAX // 10 and digit > INT_MAX % 10)):
                # If integer overflowed return 2^31-1, otherwise if underflowed return -2^31.    
                return INT_MAX if sign == 1 else INT_MIN
            
            # Append current digit to the result.
            result = 10 * result + digit
            index += 1
        
        # We have formed a valid number without any overflow/underflow.
        # Return it after multiplying it with its sign.
        return sign * result

                                                                                             215. Kth Largest Element in an Array
Given an integer array nums and an integer k, return the kth largest element in the array.
Note that it is the kth largest element in the sorted order, not the kth distinct element.
Example 1:
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
 
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # Min-Heap - return pop len(nums) - k + 1
        heapq.heapify(nums)
        output = None
        for _ in range(len(nums) - k + 1):
            output = heapq.heappop(nums)
        return output
    
        # Max-Heap - pop k - 1 times and return heappop()
        heapq._heapify_max(nums)
        for _ in range(k - 1):
            heapq._heappop_max(nums)
        return heapq._heappop_max(nums)
    
        # Use nlargest function
        return heapq.nlargest(k, nums)[-1]
    
        # Use Min-Heap and keep the size of heap to k
        heap = []
        for i in range(len(nums)):
            heapq.heappush(heap, nums[i])
            if len(heap) > k:
                heapq.heappop(heap)
            
        return heapq.heappop(heap)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
973. K Closest Points to Origin
Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).
The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).
You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).
 
Example 1:

Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
Explanation:
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        # Solution 1
        # Calculate the distance to (0, 0) and store in a heap
        # return a list of k coordinates
        
        # create a hash map with the distance to origin
        # {index : distance}
        distance_map = {}   # O(n)
        for i in range(len(points)):
            distance_map[i] = math.sqrt(points[i][0] ** 2 + points[i][1] ** 2)
        
        # put into heap
        heap = []   # O(n)
        for dist in distance_map.values():
            heapq.heappush(heap, dist)
        
        # output
        output = []
        for _ in range(k):  # O(m)
            pop = heapq.heappop(heap)
            for key, value in distance_map.items(): # O(n)
                if value == pop:
                    if points[key] not in output:
                        output.append(points[key])
        
        return output
 
        # Solution 2
        # Sort
        # Sort the list with a custom comparator function
        points.sort(key=self.squared_distance)
        
        # Return the first k elements of the sorted list
        return points[:k]
    
    def squared_distance(self, point: List[int]) -> int:
        """Calculate and return the squared Euclidean distance."""
        return point[0] ** 2 + point[1] ** 2
        
        
        # Solution 3
        heap = [(-self.squared_distance(points[i]), i) for i in range(k)]
        heapq.heapify(heap)
        for i in range(k, len(points)):
            dist = -self.squared_distance(points[i])
            if dist > heap[0][0]:
                # If this point is closer than the kth farthest,
                # discard the farthest point and add this one
                heapq.heappushpop(heap, (dist, i))
        
        # Return all points stored in the max heap
        return [points[i] for (_, i) in heap]
    
    def squared_distance(self, point: List[int]) -> int:
        """Calculate and return the squared Euclidean distance."""
        return point[0] ** 2 + point[1] ** 2
