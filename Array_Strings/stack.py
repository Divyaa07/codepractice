
# https://leetcode.com/problems/decode-string
# Input: s = "2[abc]3[cd]ef"
# Output: "abcabccdcdcdef"
# For better understanding the process, let us consider example s = 3[a5[c]]4[b]:

# [''] at first we have stack with empty string.
# ['', 3, ''], open bracket: now we have stack with 3 elements: empty string, number 3 and empty string.
# ['', 3, 'a']: build our string
# ['', 3, 'a', 5, ''], open bracket: add number and empty string
# ['', 3, 'a', 5, 'c'] build string
# ['', 3, 'accccc'] : now we have closing bracket, so we remove last 3 elements and put accccc into our stack
# ['acccccacccccaccccc'] we again have closing bracket, so we remove last 3 elements and put new one.
# ['acccccacccccaccccc', 4, '']: open bracket, add number and empty string to stack
# ['acccccacccccaccccc', 4, 'b'] build string
# ['acccccacccccacccccbbbb'] closing bracket: remove last 3 elements and put one new.

class Solution:
    def decodeString(self, s):
        it, num, stack = 0, 0, [""]
        while it < len(s):
            # For case when there is > 1 digit numbers are strings
            if s[it].isdigit():
                num = num * 10 + int(s[it])
            elif s[it] == "[":
                stack.append(num)
                num = 0
                stack.append("")
            elif s[it] == "]":
                str1 = stack.pop()
                rep = stack.pop()
                str2 = stack.pop()
                stack.append(str2 + str1 * rep)
            else:
                stack[-1] += s[it]              
            it += 1           
        return "".join(stack)


# https://www.geeksforgeeks.org/stack-set-2-infix-to-postfix/
# https://leetcode.com/problems/basic-calculator