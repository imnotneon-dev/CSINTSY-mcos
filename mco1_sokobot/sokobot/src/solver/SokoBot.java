package solver;

import java.awt.Point;
import java.util.*;

public class SokoBot {

    public String solveSokobanPuzzle(int width, int height, char[][] mapData, char[][] itemsData) {
        InitialState initialState = initializeState(width, height, mapData, itemsData);
        
        Set<Point> goals = new HashSet<>();
        // add goal points to the set
        for (int[] goalPoints : initialState.goalStates) 
            goals.add(new Point(goalPoints[0], goalPoints[1]));

        Set<Point> boxes = new HashSet<>();
        // add box points to the set
        for (int[] boxCoord : initialState.boxPoints) 
            boxes.add(new Point(boxCoord[0], boxCoord[1]));

        // initialize player starting position
        int playerX = initialState.startingPoint[0];
        int playerY = initialState.startingPoint[1];
        
        // call the GBFS solver
        GBFSAlgo solver = new GBFSAlgo();
        return solver.solve(width, height, mapData, playerX, playerY, boxes, goals);
    }

    /*
     *   HELPER FUNCTIONS
     */

    private static class InitialState {
        int[][] goalStates;
        int[] startingPoint;
        int[][] boxPoints;

        InitialState(int[][] goalStates, int[] startingPoint, int[][] boxPoints) {
            this.goalStates = goalStates;
            this.startingPoint = startingPoint;
            this.boxPoints = boxPoints;
        }
    }

    private InitialState initializeState(int width, int height, char[][] mapData, char[][] itemsData) {
        int boxCount = countBoxes(width, height, itemsData);
        int[][] goalStates = getGoalStates(width, height, mapData);
        int[] startingPoint = getCoordinates(width, height, mapData, itemsData, '@');
        int[][] boxPoints = getBoxPoints(width, height, itemsData, '$', boxCount);
        return new InitialState(goalStates, startingPoint, boxPoints);
    }

    public static int countBoxes(int width, int height, char[][] itemsData) {
        int boxes = 0;

        for (int y = 0; y < height; y++) 
            for (int x = 0; x < width; x++) 
                if (itemsData[y][x] == '$') 
                    boxes++;

        return boxes;
    }

    public static int[] getCoordinates(int width, int height, char[][] mapData, char[][] itemsData, char target) {
        for (int y = 0; y < height; y++) 
            for (int x = 0; x < width; x++)
                if (itemsData[y][x] == target || mapData[y][x] == target) 
                    return new int[]{x, y};

        return null;
    }

    public static int[][] getGoalStates(int width, int height, char[][] mapData) {
        List<int[]> goals = new ArrayList<>();

        for (int y = 0; y < height; y++) 
            for (int x = 0; x < width; x++)
                if (mapData[y][x] == '.') 
                    goals.add(new int[]{x, y});

        return goals.toArray(new int[goals.size()][]);
    }

    public static int[][] getBoxPoints(int width, int height, char[][] itemsData, char boxSymbol, int count) {
        int[][] coords = new int[count][2];
        int index = 0;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (itemsData[y][x] == boxSymbol) {
                    coords[index][0] = x;
                    coords[index][1] = y;
                    if (++index >= count) 
                        break;
                }
            }
        }

        return coords;
    }

}