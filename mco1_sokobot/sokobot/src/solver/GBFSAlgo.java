package solver;

import java.awt.Point;
import java.util.*;

public class GBFSAlgo {

    // main loop for GBFS
    public String solve(int width, int height, char[][] mapData, int playerX, int playerY, Set<Point> initialBoxes, Set<Point> goals) {

        PriorityQueue<Node> priorityQueue = new PriorityQueue<>(); // Uses Node's compareTo (based on heuristic)
        Set<String> visitedStates = new HashSet<>();
        priorityQueue.add(new Node(playerX, playerY, initialBoxes, "", goals, mapData, width, height));

        final int[][] movesDirections = { {0, -1}, {0, 1}, {-1, 0}, {1, 0} }; // up, down, left, right coor
        final char[] dirChars = { 'u', 'd', 'l', 'r' };
        final long limitStates = 2_147_483_647L;
        long statesVisited = 0;

        while (!priorityQueue.isEmpty()) {
            Node currentNode = priorityQueue.poll();
            statesVisited++;

            if (statesVisited > limitStates) {
                System.out.println("Reached max states without solution :(");
                return "";
            }
            
            // check if all boxes are on goals
            if (currentNode.boxes.equals(goals)) {
                System.out.println("Solution Found! YAY! Explored " + statesVisited + " states.");
                System.out.println("Final Path: " + currentNode.path); 
                return currentNode.path;
            }

            // check all possible moves
            String encodedState = currentNode.buildPath();
            if (visitedStates.contains(encodedState)) 
                continue;
            visitedStates.add(encodedState);

            // explore all directions
            for (int i = 0; i < movesDirections.length; i++) {
                int[] d = movesDirections[i];
                int newX = currentNode.playerX + d[0];
                int newY = currentNode.playerY + d[1];
                char dir = dirChars[i];
                
                // checks boundaries and walls
                if (newX < 0 || newY < 0 || newX >= width || newY >= height) 
                    continue;
                if (mapData[newY][newX] == '#')
                    continue;

                Set<Point> newBoxes = new HashSet<>(currentNode.boxes);
                Point nextPos = new Point(newX, newY);

                if (newBoxes.contains(nextPos)) {
                    // trying to push a box
                    int pushedX = newX + d[0];
                    int pushedY = newY + d[1];
                    Point pushedPos = new Point(pushedX, pushedY);
                    
                    // checks if box can be pushed
                    if (pushedX < 0 || pushedY < 0 || pushedX >= width || pushedY >= height) 
                        continue;
                    if (mapData[pushedY][pushedX] == '#' || newBoxes.contains(pushedPos)) 
                        continue;
                    
                    // deadlock pruning!
                    if (currentNode.isDeadlock(pushedPos))
                        continue;

                    // execute the push!
                    newBoxes.remove(nextPos);
                    newBoxes.add(pushedPos);
                }

                // create and enqueue next node
                Node nextNode = new Node(newX, newY, newBoxes, currentNode.path + dir, goals, mapData, width, height);
                String nextEncodedState = nextNode.buildPath();
                
                if (!visitedStates.contains(nextEncodedState)) {
                    priorityQueue.add(nextNode);
                }
            }
        }
        System.out.println("No solution exists.");
        return "";
    }

    // comparing nodes based on heuristic value for GBFS
    private static class Node implements Comparable<Node> {
        int playerX, playerY;
        Set<Point> boxes;
        String path;
        int heuristic;

        private final Set<Point> goals;
        private final char[][] mapData;
        private final int width, height;

        Node(int px, int py, Set<Point> boxes, String path, Set<Point> goals, char[][] mapData, int width, int height) {
            this.playerX = px;
            this.playerY = py;
            this.boxes = new HashSet<>(boxes); 
            this.path = path;
            this.goals = goals;
            this.mapData = mapData;
            this.width = width;
            this.height = height;
            this.heuristic = calculateHeuristic(this.boxes, this.goals);
        }

        @Override
        public int compareTo(Node other) {
            return Integer.compare(this.heuristic, other.heuristic);
        }

        // simple manhattan distance heuristic (sum of min distances)
        private int calculateHeuristic(Set<Point> boxSet, Set<Point> goalSet) {
            int accumulatedDistance = 0;
            for (Point box : boxSet) {
                int closestDistance = Integer.MAX_VALUE;
                for (Point goal : goalSet) {
                    int currentDistance = Math.abs(box.x - goal.x) + Math.abs(box.y - goal.y);
                    if (currentDistance < closestDistance) closestDistance = currentDistance;
                }
                accumulatedDistance += closestDistance;
            }
            return accumulatedDistance;
        }

        // encodes the state for visited checking
        public String buildPath() {
            StringBuilder sb = new StringBuilder();
            sb.append(playerX).append(',').append(playerY).append('|');
            List<Point> sortedBoxes = new ArrayList<>(boxes);
            sortedBoxes.sort(Comparator.comparingInt((Point p) -> p.x).thenComparingInt(p -> p.y));
            for (Point b : sortedBoxes) {
                sb.append(b.x).append(',').append(b.y).append(';');
            }
            return sb.toString();
        }

        // simple deadlock detection
        public boolean isDeadlock(Point box) {
            int x = box.x;
            int y = box.y;
    
            if (goals.contains(box)) {
                return false;
            }

            boolean wallUp = isWall(x, y - 1);
            boolean wallDown = isWall(x, y + 1);
            boolean wallLeft = isWall(x - 1, y);
            boolean wallRight = isWall(x + 1, y);

            // basic corner deadlock check
            if ((wallUp && wallLeft) || (wallUp && wallRight) || (wallDown && wallLeft) || (wallDown && wallRight))
                return true;

            return false;
        }

        // helper to check for walls
        private boolean isWall(int x, int y) {
            return x < 0 || y < 0 || x >= width || y >= height || mapData[y][x] == '#';
        }
    }
}