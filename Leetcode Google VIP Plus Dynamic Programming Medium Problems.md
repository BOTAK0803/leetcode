# Leetcode Google VIP Plus Dynamic Programming Medium Problems

## 276 Paint Fence

> Question: You are painting a fence of n posts with k different colors.You must paint the posts following these rules:
>
> 	- Every post must be painted exactly one color
> 	- At most one pair of adjacent fence posts can have the sanme color
>
> Given the two integers n and k,return the number of ways you can paint the fence.

```java
class Solution {
    public int numWays(int n, int k) {
        if(k == 1 && n > 2) return 0;
        if(k == 1) return 1;
        if(n==1) return k;
        // dp 动态规划
        /**
        record[i] 记录的是到第i个栅栏的时候，倒数两个是重复元素的栅栏涂色的个数
        record[0] = 0;
        record[1] = k; // 例如k=3 11 22 33
        record[i] = dp[i-1] - record[i-1]
        record[i] = 前i-1个栅栏涂完色之后的所有的情况 - 前i-1个栅栏涂完色之后倒数两个是重复元素的栅栏涂色的个数
        for example:
        k = 3,i=3
        11,12,13,21,22,23,31,32,33当算record[2]的时候，record[1] = 3，也就是11，22，33,当对下一个栅栏进行涂色的时候，
        这3种状态一定不会产生后两个是相同颜色的可能了，只有剩下的dp[i-1] - record[i-1]才有可能产生，也就是12,13,21,23,31,32
        这6中情况可以产生倒数两个是重复元素的栅栏涂色，122，133，211，233，311，322.
        状态定义 dp[i] : 表示到第i个栅栏的时候的可能的方案数
        状态转移方程 : dp[i] = (dp[i-1] - record[i-1])*k + (record[i-1])*(k-1)
        解释：当前栅栏涂完色之后的可能涂色方案数目 = (前一个栅栏涂完色的方案数目 - 前一个栅栏涂完色之后的record)* k + 前一个栅栏涂完色之后的record *（k-1） k-1表示不能涂和前一个栅栏相同的颜色了，从剩下的k-1中颜色随机挑选一个进行涂色
        初始状态:dp[0] = k; dp[1] = k*k;
        */
        int[] dp = new int[n];
        int[] record = new int[n];
        record[0] = 0;
        record[1] = k;
        dp[0] = k;
        dp[1] = k*k;
        for(int i = 2;i < n;i++){
            dp[i] = (dp[i-1] - record[i-1])*k + (record[i-1])*(k-1);
            record[i] = dp[i-1] - record[i-1];
        }
        return dp[n-1]; 

    }
}
```



## 351 Android Unlock Patterns

> Question
>
> Android devices have a special lock screen with a 3 x 3 grid of dots. Users can set an "unlock pattern" by connecting the dots in a specific sequence, forming a series of joined line segments where each segment's endpoints are two consecutive dots in the sequence. A sequence of k dots is a valid unlock pattern if both of the following are true:
>
> - All the dots in the sequence are distinct.
> - If the line segment connecting two consecutive dots in the sequence passes through any other dot, the other dot must have previously appeared in the sequence. No jumps through non-selected dots are allowed.

```java
class Solution {
    public int numberOfPatterns(int m, int n) {
        int [][]skip = new int[10][10];
        //这个skip数组是为了记录跳跃的点数，比如说从1到3，就跳跃2
        //而且因为是对称的操作，所以3到1也是如此
        skip[1][3] = skip[3][1] = 2;
        skip[1][7] = skip[7][1] = 4;
        skip[3][9] = skip[9][3] = 6;
        skip[4][6] = skip[6][4] = skip[2][8] = skip[8][2] = 5; 
        skip[1][9] = skip[9][1] = skip[3][7] = skip[7][3] = 5;
        skip[7][9] = skip[9][7] = 8;
        int result = 0;
        boolean []visited = new boolean[10];
        //深度遍历，遍历每一个点到点的次数
        for(int i = m; i<=n; i++){
            //因为从1,3,7,9出发都是对称的，为什么i要减一呢，因为我们是从1出发，先天少了一个节点
            result += DFS(1,visited,skip,i-1)*4;
            //2,4,6,8对称
            result += DFS(2,visited,skip,i-1)*4;
            //唯独5独立
            result += DFS(5,visited,skip,i-1); 
        }
        return result;
    }
    //深度遍历
    public int DFS(int current, boolean []visited, int [][]skip,int remainKeyCount){
        if(remainKeyCount == 0){
            return 1;
        }
        int result = 0;
        //深度遍历
        visited[current] = true;
        
        for(int i = 1; i <= 9; i++){
            //看当前的节点到i节点的路径中有没有其他节点在中间
              int crossThroughNumber = skip[current][i];
              //如果这一次我们的i节点没有被读过，那么就判断有没有路过中间节点(visited[crossThroughNumber])或者这两个节点相邻没有中间节点（currentThrough=0）
              if(!visited[i] && (crossThroughNumber == 0 ||visited[crossThroughNumber])){
                 result += DFS(i,visited,skip,remainKeyCount-1); 
              }
        }
        //渣男行径开始了
        visited[current] = false;
        return result;
    }
}
```



## 361 Bomb Enemy

> Question
>
> Given an m x n matrix grid where each cell is either a wall 'W', an enemy 'E' or empty '0', return the maximum enemies you can kill using one bomb. You can only place the bomb in an empty cell.
>
> The bomb kills all the enemies in the same row and column from the planted point until it hits the wall since it is too strong to be destroyed.

```java
class Solution {
    int[][] f1, f2, f3, f4;
    public int maxKilledEnemies(char[][] grid){
        int r = grid.length; 
        if(r == 0)  return 0;
        int c = grid[0].length;
        if(c == 0)  return 0;
        f1 = new int[r + 2][c + 2]; f2 = new int[r + 2][c + 2];
        f3 = new int[r + 2][c + 2]; f4 = new int[r + 2][c + 2];
        for(int i = 1; i <= r; i++){
            for(int j = 1; j <= c; j++){
                f1[i][j] = f1[i][j - 1];
                if(grid[i - 1][j - 1] == 'E') f1[i][j]++;
                else if(grid[i - 1][j - 1] == 'W') f1[i][j] = 0;
            }

            for(int j = c; j >= 1; j--){
                f2[i][j] = f2[i][j + 1];
                if(grid[i - 1][j - 1] == 'E') f2[i][j]++;
                else if(grid[i - 1][j - 1] == 'W') f2[i][j] = 0;
            }
        }
        int ans = 0;
        for(int j = 1; j <= c; j++){
            for(int i = 1; i <= r; i++){
                f3[i][j] = f3[i - 1][j];
                if(grid[i - 1][j - 1] == 'E') f3[i][j]++;
                else if(grid[i - 1][j - 1] == 'W') f3[i][j] = 0;
            }

            for(int i = r; i >= 1; i--){
                f4[i][j] = f4[i + 1][j];
                if(grid[i - 1][j - 1] == 'E') f4[i][j]++;
                else if(grid[i - 1][j - 1] == 'W') f4[i][j] = 0;
                if(grid[i - 1][j - 1] == '0'){
                    ans = Math.max(ans, f1[i][j] + f2[i][j] + f3[i][j] + f4[i][j]);
                }
            }
        }
        return ans;
    }
}
```



## 418 Sentence Screen Fitting

> Question
>
> Given a rows x cols screen and a sentence represented as a list of strings,return the number of times the given sentence can be fitted on the screen.
>
> The order of words in the sentence must remain unchanged, and a word cannot be split into two lines. A single space must separate two consecutive words in a line.

```java
class Solution {
    public int wordsTyping(String[] sentence, int rows, int cols) {
        int senLen = sentence.length;
        int result = 0;
        // 每行剩余的容量，初始值为列数
        int rowLeft = cols;
        // 数组中的第几个单词
        int wordIndex = 0;
        int rowIndex = 0;
        while (rowIndex < rows) {
            int worldLength = sentence[wordIndex].length();
            // 一行可以放下就继续放
            if (rowLeft >= worldLength) {
                rowLeft = rowLeft - worldLength;
                // 空格
                rowLeft--;
                wordIndex++;
                // 放完一个句子重新开始
                if (wordIndex == senLen) {
                    result++;
                    wordIndex = 0;
                }
            }
            // 一行放不下就换行
            else {
                rowIndex++;
                rowLeft = cols;
            }
        }
        return result;
    }
}
```



## 651 4 Keys Keyboard

> Question
>
> Imagine you have a special keyboard with the following keys:
>
> Key 1 : (A) : Print one 'A' on screen
>
> Key 2 : (Ctrl-A) : Select the whole screen
>
> Key 3 :(Ctrl-C) : Copyselection to buffer.
>
> Key 4 :(Ctrl-V) : Print buffer on screen appending it after what has already been printed.
>
> Now , you can only press the keyboard for N times (with the above four keys),find out maximum numbers of 'A' you can print on screen.

贪心：最后一步，要么是A，要么是Ctrl-V

State:

dp[i]表示到第i次操作，屏幕上能显示A的最大个数

Function:

dp[i] = Math.max(dp[i-1] + 1,dp[i - j - 1]) 

假设从第j步执行Ctrl-A,第j+1步执行Ctrl-C，那么剩下的i-(j+1)步执行Ctrl-V 所以dp[i] = max(dp[j] * (i-j-1))

Initial State:

dp[i] = i;

```java
class Solution{
    public int maxA(int n){
        int[] dp = new int[n+1];
        for(int i = 1;i<=n;i++){
            dp[i] = dp[i-1]+1;
            for(int j = 2;j+2<i;j++){
                dp[i] = Math.max(dp[i],dp[j]*(i-j+1));
            }
        }
        return dp[n];
        
    }
}
```



## 1055 Shortest Way to Form String

> Question:
>
> From any string , we can form a subsequence of that string by deleting some number of characters (possibly no deletions)
>
> Given two strings source and target.return the minimum number of subsequences of source such that their concatenation equals target.if the task is impossible , return -1;

```java
class Solution{
    public int shortestWay(String source,String target){
        int res = 0;
        int s = 0;
        int t = 0;
        while(t<target.length()){
            if(source.indexOf(target.charAt(t)) == -1){
                return -1;
            }
            if(source.charAt(s) == target.charAt(t)){
                s++;
                t++;
            }else{
                s++;
            }
            if(s>=source.length()){
                res++;
                s=0;
            }
        }
        if(t>=target.length() && s!=0) res++;
        return res;
    }
}
```



## 1066 Campus Bikes II

> Question
>
> On a campus represented as a 2D grid, there are N workers and M bikes, with N <= M. Each worker and bike is a 2D coordinate on this grid.
>
> We assign one unique bike to each worker so that the sum of the Manhattan distances between each worker and their assigned bike is minimized.
>
> The Manhattan distance between two points p1 and p2 is Manhattan(p1, p2) = |p1.x - p2.x| + |p1.y - p2.y|.
>
> Return the minimum possible sum of Manhattan distances between each worker and their assigned bike.

```java
class Solution {
    int minDistance = Integer.MAX_VALUE;
    public int assignBikes(int[][] workers, int[][] bikes) {
        return helper(workers, bikes, new int[bikes.length], 0);
    }

    /**
     *
     * @param workers
     * @param bikes
     * @param visited   代表访问过的自行车
     * @param index     代表当前遍历到的工人
     * @return
     */
    public int helper(int[][] workers, int[][] bikes, int[] visited, int index) {
        if (index == workers.length) {
            return 0;
        }

        for (int i = 0; i < bikes.length; i++) {
            if (visited[i] == 0) {
                int distance = Manhattan(workers[index][0], workers[index][1], bikes[i][0], bikes[i][1]);
                visited[i] = 1;
                minDistance = Math.min(minDistance, distance + helper(workers, bikes, visited, index + 1));
                visited[i] = 0;
            }
        }
        return minDistance;
    }

    public int Manhattan(int x1, int y1, int x2, int y2) {
        return Math.abs(x1 - x2) + Math.abs(y1 - y2);
    }
}
```



## 1136 Parallel Courses

> Question
>
> You are given an integer n which indicates that we have n courses, labeled from 1 to n. You are also given an array relations where relations[i] = [a, b], representing a prerequisite relationship between course a and course b: course a has to be studied before course b.
>
> In one semester, you can study any number of courses as long as you have studied all the prerequisites for the course you are studying.
>
> Return the minimum number of semesters needed to study all courses. If there is no way to study all the courses, return -1.

```java
class Solution {
    public int minimumSemesters(int N, int[][] relations) {
        int[] degree = new int[N + 1];
        //构建图的数据结构，给定课程的关系，有后继课程的其入度+1
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] r : relations) {
            int a = r[0];
            int b = r[1];
            //b是a的后继课程，所以入度+1
            degree[b]++;
            //a课程是否存在，不存在就给它新建一个
            if (graph.containsKey(a)) {
                graph.get(a).add(b);
            } else {
                List<Integer> list = new LinkedList<>();
                list.add(b);
                graph.put(a, list);
            }
        }
//        新学期要上的课程，就是入度为0的课程放进去
        List<Integer> nowCourse = new LinkedList<>();
        for (int i = 1; i < degree.length; i++) {
            if (degree[i] == 0) {
                nowCourse.add(i);
            }
        }
        //最小学期数
        int mini = 0;
//        开始上课，开学了
        while (!nowCourse.isEmpty()) {
//            每次新学期开始前，要新建下一学期的nextCourse
            List<Integer> nextCourse = new LinkedList<>();
//            遍历要上的课
            for (int c : nowCourse) {
//                如果在总的课程安排里面，它的后继课程不为空
                if (graph.get(c) != null) {
//                    不为空，就获得它的全部后继课程，赋值给y
                    List<Integer> y = graph.get(c);
//                    然后遍历y，并入度-1，并统计有没有0，加入到下学期课中
                    for (Integer a : y) {
                        degree[a]--;
                        if (degree[a] == 0) {
                            nextCourse.add(a);
                        }
                    }
                }
            }
//            学期结束
            mini++;
//            下学期的课给到新学期，开始新的学期
            nowCourse = nextCourse;
        }
//        统计入度是否不为0的，有则存在环，返回-1
        for (int i = 1; i < degree.length; i++) {
            if (degree[i] > 0) {
                return -1;
            }
        }
        return mini;
    }
}

```



## 256 Paint House

> Question
>
> There is a row of n houses, where each house can be painted one of three colors: red, blue, or green. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.
>
> The cost of painting each house with a certain color is represented by an n x 3 cost matrix costs.

$dp[i][j]$: 表示粉刷完第i个房子用第j种颜色的最小花费

```java
class Solution {
    public int minCost(int[][] costs) {
        int n = costs.length;
        if (n == 0) return 0;
        int[][] dp = new int[n][3];
        dp[0][0] = costs[0][0];
        dp[0][1] = costs[0][1];
        dp[0][2] = costs[0][2];
        for (int i = 1; i < n; i++) {
            dp[i][0] = Math.min(dp[i - 1][1], dp[i - 1][2]) + costs[i][0];
            dp[i][1] = Math.min(dp[i - 1][2], dp[i - 1][0]) + costs[i][1];
            dp[i][2] = Math.min(dp[i - 1][0], dp[i - 1][1]) + costs[i][2];
        }
        return Math.min(Math.min(dp[n - 1][0], dp[n - 1][1]), dp[n - 1][2]);
    }
}
```



## 750 Number Of Corner Rectangles

> Question
>
> Given a grid where each entry is only 0 or 1, find the number of corner rectangles.
>
> A corner rectangle is 4 distinct 1s on the grid that form an axis-aligned rectangle. Note that only the corners need to have the value 1. Also, all four 1s used must be distinct.

```java
class Solution {
    public int countCornerRectangles(int[][] grid) {
        int rows = grid.length;
        int cols = grid[0].length;
        // border[i][j] 表示从不包含当前行的前面所有行中，列的 i 索引和 j 索引都为 1 的行的个数
        int[][] border = new int[rows + 1][cols + 1];
        int ans = 0;    // 总共角矩形的个数
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 1) {
                    for (int k = j + 1; k < cols; k++) {
                        if (grid[i][k] == 1) {  // 找到了角矩形的一个底边
                            ans += border[j][k];
                            // 第 0 行到第 i 行的每一行中，列索引 j 和 k 都为 1 的行的个数加一
                            border[j][k]++;
                        }
                    }
                }
            }
        }
        return ans;
    }
}
```



## 1273 Delete Tree Nodes

> Question:
>
> A tree rooted at node 0 is given as follows
>
> 	- The number of nodes is nodes;
> 	- The value of the i-th node is value[i]
> 	- The parent of the i-th node is parent[i]
>
> Remove every subtree whose sum of values of nodes is zero;
>
> After doing that , return the number of nodes remaining in the tree.

```java
class Solution{
    static class TreeNode{
        int value;
        int sum = null;
        int num = null;
        List<TreeNode> children;
        public TreeNode(){
            this.children = new ArrayList<TreeNode>();
        }
        public TreeNode(int value){
            this.value = value;
            this.children = new ArrayList<TreeNode>();
        }
        public void addTreeNode(TreeNode node){
            this.children.add(node);
        }
    }
    public int deleteTreeNodes(int nodes,int[] parent,int[] value){
        TreeNode root = null;
        TreeNode[] treeArr = new TreeNode[nodes];
        for(int i = 0;i<nodes;i++){
            treeArr[i] = new TreeNode(value[i]);
            if(parent[i] == -1) root = treeArr[i];
        }
        for(int i = 0;i< nodes;i++){
            if(parents[i] != -1){
                treeArr[parents[i]].addTreeNode(TreeArr[i]);
            }
        }
        computeTree(root);
        if(root.sum == 0) return 0;
        return nodes - removeZeroTreeNode(root);
    }
    public void computeTree(TreeNode node){
        node.sum = node.value;
        node.num = 1;
        for(TreeNode child:node.children){
            computeTree(child);
            node.sum+=child.sum;
            node.num+=child.num;
        }
    }
    public int removeZeroTreeNode(TreeNode node){
        int removeNum = 0;
        for(int i = 0;i< node.children.size();i++){
            TreeNode child = node.children.get(i);
            if(child.sum == 0){
                removeNum += child.num;
            }else{
                removeNum += removeZeroTreeNode(child);
            }
        }
        return removeNum;
    }
}
```



## 1230 Toss Strange Coins

> Question:
>
> You have some coins.The i-th coin has a probability prob[i] of facing heads when tossed.
>
> Return the probability that the number of coins facing heads equals target . if you toss every coin exactly once. 

State:

$dp[i][j] $：i个硬币投出j个正片朝上的概率值

Function：

用i个硬币投出j个正面朝上的概率取决于用i-1个硬币投出多少个正面朝上的概率。

如果i-1个硬币投出了j-1个正面朝上，那么第i个硬币一定要投出正面朝上。

如果i-1个硬币已经投出了j个正面朝上，那么第i个硬币就要投出反面朝上。

$dp[i][j] = dp[i-1][j] * (1-prob[i] + dp[i-1][j-1] * prob[i])$

Initial State:

$dp[i][0] = dp[i-1][0] * (1-prob[i-1])$.

```java
class Solution{
    public double probabilityOfHeads(double[] prob,int target){
        int n = prob.length;
        double[][] dp = new double[n+1][target+1];
        dp[0][0] = 1;
        for(int i = 1;i<=n;i++) dp[i][0] = dp[i-1][0] * (1-prob[i-1]);
        for(int i = 1;i<=n;i++){
            for(int j = 1;j<=target;j++){
                dp[i][j] = dp[i-1][j-1] * prob[i-1] + dp[i-1][j] * (1 - prob[i-1]);
            }
        }
        return dp[n][target];
    }
}
```







