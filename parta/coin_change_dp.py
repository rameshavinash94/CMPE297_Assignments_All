#coin change problem
def coin_change_dp(coins, amount):
    dp = [0] + [float('inf')] * amount
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[-1] if dp[-1] != float('inf') else -1


coins = [1, 2, 5]
amount = 11
print(coin_change_dp(coins, amount))

# coin change problem with memoization
def coin_change_dp_memo(coins, amount):
    dp = [0] + [float('inf')] * amount
    def helper(coins, amount, dp):
        if amount == 0:
            return 0
        if dp[amount] != float('inf'):
            return dp[amount]
        for coin in coins:
            if coin <= amount:
                dp[amount] = min(dp[amount], helper(coins, amount - coin, dp) + 1)
        return dp[amount]
    return helper(coins, amount, dp)


coins = [1, 2, 5]
amount = 11
print(coin_change_dp_memo(coins, amount))

# coin change problem with tabulation
def coin_change_dp_tab(coins, amount):
    dp = [0] + [float('inf')] * amount
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[-1] if dp[-1] != float('inf') else -1


coins = [1, 2, 5]
amount = 11
print(coin_change_dp_tab(coins, amount))

# coin change problem with tabulation and memoization
def coin_change_dp_tab_memo(coins, amount):
    dp = [0] + [float('inf')] * amount
    def helper(coins, amount, dp):
        if amount == 0:
            return 0
        if dp[amount] != float('inf'):
            return dp[amount]
        for coin in coins:
            if coin <= amount:
                dp[amount] = min(dp[amount], helper(coins, amount - coin, dp) + 1)
        return dp[amount]
    return helper(coins, amount, dp)


coins = [1, 2, 5]
amount = 11
print(coin_change_dp_tab_memo(coins, amount))

# coin change problem with tabulation and memoization and bottom up

def coin_change_dp_tab_memo_bottomup(coins, amount):
    dp = [0] + [float('inf')] * amount
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[-1] if dp[-1] != float('inf') else -1


coins = [1, 2, 5]
amount = 11
print(coin_change_dp_tab_memo_bottomup(coins, amount))
