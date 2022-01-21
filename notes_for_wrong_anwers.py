'''
Delete duplicates in a list
'''
list = [1, 1, 2, 3, 4, 5, 5, 5, 6, 7]
# number_count = []
# for i in list:
#     list.count(i)
#     number_count.append(i)
# index = 0
# for j in number_count:
#     if j > 1:
#         list.remove(list.find(index))
#         index += 1
unique = []
for number in list:
    if number not in unique:
        unique.append(number)
print(unique)



'''
Let’s print a triangle made of asterisks (‘*’) separated by spaces and consisting
of n rows again, but this time upside down, and make it symmetrical. Consecutive rows should
contain 2n − 1, 2n − 3, . . . , 3, 1 asterisks and should be indented by 0, 2, 4, . . . , 2(n − 1)
spaces.
'''
def print_stars_triangle(n):
    for i in range(n, 0, -1):
        for j in range(n -i):
            print(' ', end='')
        for j in range(2 * i - 1):
            print('*', end='')
        print('')

number = int(input('number: '))
print_stars_triangle(number)



'''
Count the number of character in a string and store it in a dictionary
'''
def count_characters_in_string(string):
    output = {}
    for char in string:
        if char in output:
            output[char] += 1
        else:
            output[char] = 1
    
    return output


'''
## Codility Test 6: TapeEquilibrium

A non-empty array A consisting of N integers is given. Array A represents numbers on a tape.

Any integer P, such that 0 < P < N, splits this tape into two non-empty parts: A[0], A[1], ..., A[P − 1] and A[P], A[P + 1], ..., A[N − 1].

The difference between the two parts is the value of: |(A[0] + A[1] + ... + A[P − 1]) − (A[P] + A[P + 1] + ... + A[N − 1])|

In other words, it is the absolute difference between the sum of the first part and the sum of the second part.
'''
def solution(A):
    if len(A) < 2:
        return 0
    
    total_sum = sum(A)
    min_differnce = 2000

    left_sum = 0
    for i in range(len(A) - 1):
        left_sum += A[i]
        differnce = abs(2 * left_sum - total_sum)
        min_differnce = min(min_differnce, differnce)
    
    return min_differnce



'''
## Codility Test 7: FrogRiverOne

A small frog wants to get to the other side of a river. The frog is initially located on one bank of the river (position 0) and wants to get to the opposite bank (position X+1). Leaves fall from a tree onto the surface of the river.

You are given an array A consisting of N integers representing the falling leaves. A[K] represents the position where one leaf falls at time K, measured in seconds.

The goal is to find the earliest time when the frog can jump to the other side of the river. The frog can cross only when leaves appear at every position across the river from 1 to X (that is, we want to find the earliest moment when all the positions from 1 to X are covered by leaves). You may assume that the speed of the current in the river is negligibly small, i.e. the leaves do not change their positions once they fall in the river.

For example, you are given integer X = 5 and array A such that:

  A[0] = 1
  A[1] = 3
  A[2] = 1
  A[3] = 4
  A[4] = 2
  A[5] = 3
  A[6] = 5
  A[7] = 4
In second 6, a leaf falls into position 5. This is the earliest time when leaves appear in every position across the river.

Write a function:

def solution(X, A)

that, given a non-empty array A consisting of N integers and integer X, returns the earliest time when the frog can jump to the other side of the river.

If the frog is never able to jump to the other side of the river, the function should return −1.
'''
def solution(X, A):
    leaves_position = [0] * X
    number_of_leaves = 0

    for i in range(len(A)):
        A[i] -= 1

    for i in range(len(A)):
        if leaves_position[A[i]] != 1:
            leaves_position[A[i]] = 1
            number_of_leaves += 1
        if number_of_leaves == X:
            return i

    return -1



'''
## Codility Test 10: MissingInteger

Write a function:

def solution(A)

that, given an array A of N integers, returns the smallest positive integer (greater than 0) that does not occur in A.

For example, given A = [1, 3, 6, 4, 1, 2], the function should return 5.

Given A = [1, 2, 3], the function should return 4.

Given A = [−1, −3], the function should return 1.
'''
def solution(A):
    container = list(range(1, max(A) + 1))

    for i in range(len(A)):
        try:
            container.remove(A[i])
        except(ValueError):
            continue
    
    if max(A) < 1:
        return 1
    elif not container:
        return max(A) + 1
    else:
        return container[0]



'''
## Codility Test 11: PassingCars

A non-empty array A consisting of N integers is given. The consecutive elements of array A represent consecutive cars on a road.

Array A contains only 0s and/or 1s:

0 represents a car traveling east,
1 represents a car traveling west.
The goal is to count passing cars. We say that a pair of cars (P, Q), where 0 ≤ P < Q < N, is passing when P is traveling to the east and Q is traveling to the west.

For example, consider array A such that:

  A[0] = 0
  A[1] = 1
  A[2] = 0
  A[3] = 1
  A[4] = 1
We have five pairs of passing cars: (0, 1), (0, 3), (0, 4), (2, 3), (2, 4).
'''
def solution(A):
    travelling_east = 0
    passing_cars = 0

    for i in range(len(A)):
        if A[i] == 0:
            travelling_east += 1
        elif A[i] == 1:
            passing_cars += travelling_east
    
    if passing_cars > 1000000000:
        return -1
    else:
        return passing_cars

'''
## Codility Test 12: GenomicRangeQuery

A DNA sequence can be represented as a string consisting of the letters A, C, G and T, which correspond to the types of successive nucleotides in the sequence. Each nucleotide has an impact factor, which is an integer. Nucleotides of types A, C, G and T have impact factors of 1, 2, 3 and 4, respectively. You are going to answer several queries of the form: What is the minimal impact factor of nucleotides contained in a particular part of the given DNA sequence?

The DNA sequence is given as a non-empty string S = S[0]S[1]...S[N-1] consisting of N characters. There are M queries, which are given in non-empty arrays P and Q, each consisting of M integers. The K-th query (0 ≤ K < M) requires you to find the minimal impact factor of nucleotides contained in the DNA sequence between positions P[K] and Q[K] (inclusive).

For example, consider string S = CAGCCTA and arrays P, Q such that:

    P[0] = 2    Q[0] = 4
    P[1] = 5    Q[1] = 5
    P[2] = 0    Q[2] = 6
The answers to these M = 3 queries are as follows:

The part of the DNA between positions 2 and 4 contains nucleotides G and C (twice), whose impact factors are 3 and 2 respectively, so the answer is 2.
The part between positions 5 and 5 contains a single nucleotide T, whose impact factor is 4, so the answer is 4.
The part between positions 0 and 6 (the whole string) contains all nucleotides, in particular nucleotide A whose impact factor is 1, so the answer is 1.
'''
def solution(S, P, Q):
    # output = []
    # impact_factors = []

    # for i in range(len(S)):
    #     if S[i] == 'A':
    #         impact_factors.append(1)
    #     elif S[i] == 'C':
    #         impact_factors.append(2)
    #     elif S[i] == 'G':
    #         impact_factors.append(3)
    #     elif S[i] == 'T':
    #         impact_factors.append(4)
    #     else:
    #         pass

    # for i in range(len(P)):
    #     target = impact_factors[P[i] : Q[i] + 1]
    #     output.append(min(target))

    # return output

    # 1. Convert S to cumulative sum lists for each A, C, G, and T.
    # 2. Check if the values of P and Q are same -> return the impact factor of it.
    # 3. else: Check if there's any differece between two values -> meaning corresponding letters have occured.

    cumul_sum_A = [0] * len(S)
    cumul_sum_C = [0] * len(S)
    cumul_sum_G = [0] * len(S)
    cumul_sum_T = [0] * len(S)

    output = []

    for i in range(len(S)):
        if S[i] == 'A':
            cumul_sum_A[i] = 1
        elif S[i] == 'C':
            cumul_sum_C[i] = 1
        elif S[i] == 'G':
            cumul_sum_G[i] = 1
        elif S[i] == 'T':
            cumul_sum_T[i] = 1
    
    
    for i in range(len(S)):
        try:
            cumul_sum_A[i + 1] += cumul_sum_A[i]
            cumul_sum_C[i + 1] += cumul_sum_C[i]
            cumul_sum_G[i + 1] += cumul_sum_G[i]
            cumul_sum_T[i + 1] += cumul_sum_T[i]
        except(IndexError):
            continue

    print(cumul_sum_A)
    print(cumul_sum_C)
    print(cumul_sum_G)
    print(cumul_sum_T)

    for i in range(len(P)):
        if P[i] == Q[i]:
            if S[P[i]] == 'A':
                output.append(1)
            elif S[P[i]] == 'C':
                output.append(2)
            elif S[P[i]] == 'G':
                output.append(3)
            elif S[P[i]] == 'T':
                output.append(4)
            # if cumul_sum_A[P[i]] != 0:
            #     output.append(1)
            # elif cumul_sum_C[P[i]] != 0:
            #     output.append(2)
            # elif cumul_sum_G[P[i]] != 0:
            #     output.append(3)
            # elif cumul_sum_T[P[i]] != 0:
            #     output.append(4)
        else:
            if P[i] == 0:
                if cumul_sum_A[P[i]] != cumul_sum_A[Q[i]]:
                    output.append(1)
                elif cumul_sum_C[P[i]] != cumul_sum_C[Q[i]]:
                    output.append(2)
                elif cumul_sum_G[P[i]] != cumul_sum_G[Q[i]]:
                    output.append(3)
                elif cumul_sum_T[P[i]] != cumul_sum_T[Q[i]]:
                    output.append(4)
            else:
                if cumul_sum_A[P[i] - 1] != cumul_sum_A[Q[i]]:
                    output.append(1)
                elif cumul_sum_C[P[i] - 1] != cumul_sum_C[Q[i]]:
                    output.append(2)
                elif cumul_sum_G[P[i] - 1] != cumul_sum_G[Q[i]]:
                    output.append(3)
                elif cumul_sum_T[P[i] - 1] != cumul_sum_T[Q[i]]:
                    output.append(4)
    
    return output


'''
HackerRank Mock Test - Flipping the matrix
'''
def flipping_matrix(matrix):
    n = len(matrix)
    s = 0
    for i in range(n//2):
        for j in range(n//2):
            s += max(matrix[i][j], matrix[i][n-j-1], matrix[n-i-1][j], matrix[n-i-1][n-j-1])
    return s



'''
LeetCode - Duplicate Zeors
'''
def duplicateZeros(self, arr: List[int]) -> None:
        zeroes = arr.count(0)
        n = len(arr)
        for i in range(n-1, -1, -1):
            if i + zeroes < n:
                arr[i + zeroes] = arr[i]
            if arr[i] == 0: 
                zeroes -= 1
                if i + zeroes < n:
                    arr[i + zeroes] = 0
                    


'''
LeetCode - Remove Duplicates from Sorted Array (Solution in bubble compare way)
'''
def removeDuplicates(self, nums: List[int]) -> int:
        # appeared = []
        # length = len(nums)
        # start = 0
        # end = length - 1
        # while start <= end:
        #     # print(f'start: {start}, end: {end}, nums[start]: {nums[start]}, appeared: ', end='')
        #     # print(appeared)
        #     if nums[start] in appeared:
        #         # print(f'deleted: {nums[start]}, ', end='')
        #         del nums[start]
        #         # print('nums: ', end='')
        #         # print(nums)
        #         end -= 1
        #     else:
        #         appeared.append(nums[start])
        #         start += 1
        if not nums:
            return 0
        
        index = 1
        n = len(nums)
        for i in range(n-1):
            if nums[i] != nums[i+1]:
                nums[index] = nums[i+1]
                index += 1
        return index
    
    
    
'''
LeetCode - Replace Elements with Greatest Element on Right Side
'''
def replaceElements(self, arr: List[int]) -> List[int]:
        mx = -1
        length = len(arr)
            
        for i in range(length - 1, -1, -1):
            tmp = arr[i]
            arr[i] = mx
            mx = max(mx, tmp)
            
        return arr
    
    
#         output = []
#         length = len(arr)
#         for i in range(1, length):
#             output.append(max(arr[i:]))
#         output.append(-1)
        
#         return output

'''
LeetCode - Singly-Linked List
'''
class ListNode:
    
    def __init__(self, val):
        self.val = val
        self.next = None
        

class MyLinkedList:

    def __init__(self):
        '''
        Initializes the MyLinkedList object.
        '''
        
        self.head = None
        self.size = 0

    def get(self, index: int) -> int:
        '''
        Get the value of the indexth node in the linked list. If the index is invalid, return -1.
        '''
        if index < 0 or index > self.size - 1:
            return -1
        
        current = self.head
        
        for _ in range(index):
            current = current.next
            
        return current.val
            

    def addAtHead(self, val: int) -> None:
        '''
        Add a node of value val before the first element of the linked list. 
        After the insertion, the new node will be the first node of the linked list.
        '''
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        '''
        Append a node of value val as the last element of the linked list.
        '''
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index: int, val: int) -> None:
        '''
        Add a node of value val before the indexth node in the linked list. 
        If index equals the length of the linked list, 
        the node will be appended to the end of the linked list. 
        If index is greater than the length, the node will not be inserted.
        '''
        if index > self.size:
            return
        
        current = self.head
        new_node = ListNode(val)
        
        if index <= 0:
            new_node.next = current
            self.head = new_node
        else:
            for _ in range(index - 1):
                current = current.next
            new_node.next = current.next
            current.next = new_node
        
        self.size += 1
        

    def deleteAtIndex(self, index: int) -> None:
        '''
        Delete the indexth node in the linked list, if the index is valid.
        '''
        if index < 0 or index >= self.size:
            return
        
        current = self.head
        
        if index == 0:
            self.head = self.head.next
        else:
            for _ in range(index - 1):
                current = current.next
            current.next = current.next.next
        
        self.size -= 1
        
'''Tips!!!! Singly Linked List Algorithms: Two Pointer, Sentinel Node'''




'''
Binary Tree Traversal - Depth First Search - Preorder Traversal
'''
def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    output = []
    self.helper(root, output)
    
    return output

def helper(self, root, output):
    if root:
        output.append(root.val)
        self.helper(root.left, output)
        self.helper(root.right, output)


'''
Binary Tree Traversal - Depth First Search - Inorder Traversal
'''
def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    output = []
    self.helper(root, output)
    
    return output

def helper(self, root, output):
    if root:
        self.helper(root.left, output)
        output.append(root.val)
        self.helper(root.right, output)
        

'''
Binary Tree Traversal - Depth First Search - Preorder Traversal
'''
def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
    output = []
    self.helper(root.output)
    
    return output

def helper(self, root, output):
    if root:
        self.helper(root.left, output)
        self.helper(root.right, output)
        output.append(root.val)
        

'''
Binary Tree Traversal - Breadth First Seach - Level-order Traversal
'''
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    levels = []
    if not root:
        return levels
    
    self.helper(root, 0, levels)
    
    return levels

def helper(self, node, level, levels):
    if len(levels) == level:
        levels.append([])
    
    levels[level].append(node.val)
    
    if node.left:
        self.helper(node.left, level + 1, levels)
    if node.right:
        self.helper(node.right, level + 1, levels)
    