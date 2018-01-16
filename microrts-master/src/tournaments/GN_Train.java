package tournaments;

import rts.PhysicalGameState;
import rts.units.Unit;

import java.util.ArrayList;

/**
 * Created by admin on 16/01/2018.
 */
public class GN_Train {
    private static final int NUM_OF_BITS = 11;

    final static int RESOURCE = 0;
    final static int BASE = RESOURCE+1;
    final static int BARRACKS = RESOURCE+2;
    final static int WORKER = RESOURCE+3;
    final static int LIGHT = RESOURCE+4;
    final static int HEAVY = RESOURCE+5;
    final static int RANGED = RESOURCE+6;
    final static int WALL = RESOURCE+7;
    final static int EMPTY = RESOURCE+8;
    final static int PLAYER0 = RESOURCE+9;
    final static int PLAYER1 = RESOURCE+10;
    final static int MAP_SIZE = 32;

    public static void main(String[] args)
    {
        String[][][] map = new String[][][]{
                {{"1"},{"1"},{"1"},{"1"}},
                {{"2"},{"2"},{"2"},{"2"}},
                {{"3"},{"3"},{"3"},{"3"}},
                {{"4"},{"4"},{"4"},{"4"}},};
       // rotateInPlace90DegreesClockwise(map);
       // rotateInPlace90DegreesClockwise(map);
       // rotateInPlace90DegreesClockwise(map);
        for(int i=0;i<map.length;i++) {
            for (int j = 0; j < map[i].length; j++)
                System.out.print(map[i][j][0]);
            System.out.println();
        }


    }
    public static void rotateInPlace90DegreesClockwise(int[][][] matrix) {
        int n = matrix.length;
        int half = n / 2;

        for (int layer = 0; layer < half; layer++) {
            int first = layer;
            int last = n - 1 - layer;

            for (int i = first; i < last; i++) {
                int offset = i - first;
                int j = last - offset;
                int[] top = matrix[first][i]; // save top

                // left -> top
                matrix[first][i] = matrix[j][first];
                // bottom -> left
                matrix[j][first] = matrix[last][j];
                // right -> bottom
                matrix[last][j] = matrix[i][last];
                // top -> right
                matrix[i][last] = top; // right <- saved top
            }
        }
    }

    public static void sendSnapshots(ArrayList<Snapshot> snapshots) {
        ArrayList<NeuralInput> inpArr = new ArrayList<>();
        for(Snapshot snp : snapshots)
        {
            int [][][] map = new int[snp.state.getWidth()][snp.state.getHeight()][NUM_OF_BITS];
            initMap(map);
            for(int i=0;i<snp.state.getWidth();i++)
                for(int j=0;j<snp.state.getHeight();j++) {
                    if(snp.state.getTerrain(j,i)==1)
                        map[i][j][WALL] = 1;//todo maybe to delete
                    else
                        map[i][j][EMPTY] = 1;
                }

            for(Unit unit : snp.state.getUnits())
            {

                String UnitID = unit.getType().name;
                insertToMap(snp.state,map,unit.getX(),unit.getY(),UnitID,unit.getPlayer());
            }

            addInstance(inpArr, snp, map, false);
            addInstance(inpArr, snp, rotateMap(map),true);
            return;
        }
        CNN_INPUT(inpArr);
    }

    private static void addInstance(ArrayList<NeuralInput> inpArr, Snapshot snp, int[][][] map, boolean isFlipped) {
        int[][][] newMap = padMap(copyMap(map));
        NeuralInput n1 = new NeuralInput();
        n1.arr= newMap;
        n1.p1=snp.p1;
        n1.p2=snp.p2;
        n1.idx = snp.snapIDX;
        if(isFlipped){
            System.out.println("p1="+snp.p2);
            System.out.println("p2="+snp.p1);
        }
        else{
            System.out.println("p1="+snp.p1);
            System.out.println("p2="+snp.p2);
        }
        //printMap(newMap);
        inpArr.add(n1);
    }

    private static int[][][] rotateMap(int[][][] map) {
        int[][][] newMap = copyMap(map);
        rotateInPlace90DegreesClockwise(newMap);
        rotateInPlace90DegreesClockwise(newMap);
        return newMap;
    }

    private static int[][][] padMap(int[][][] map) {
        int [][][] newMap = new int[MAP_SIZE][MAP_SIZE][NUM_OF_BITS];
        int[] wall = new int[NUM_OF_BITS];
        wall[WALL]=1;
        for(int i=0;i<newMap.length;i++)
            for(int j=0;j<newMap[i].length;j++)
                for(int k=0;k<newMap[i][j].length;k++)
                    System.arraycopy( wall, 0, newMap[i][j], 0, wall.length );

        for(int i=0;i<map.length;i++)
            for(int j=0;j<map[i].length;j++)
                for(int k=0;k<map[i][j].length;k++)
                    newMap[i][j][k]=map[i][j][k];
        return newMap;
    }

    private static void printMap(int[][][] map) {
        for(int i=0;i<map.length;i++)
            for(int j=0;j<map[i].length;j++) {
                System.out.print(i+","+j+",");
                for (int k = 0; k < map[i][j].length; k++)
                    System.out.print(map[i][j][k]);
                System.out.println();
            }
    }
    private static int[][][] copyMap(int[][][] map)
    {
        int[][][] newArr = new int[map.length][map[0].length][map[0][0].length];
        for(int i=0; i<map.length; i++)
            for(int j=0; j<map[i].length; j++)
                for(int k=0; k<map[i][j].length; k++)
                    newArr[i][j][k]=map[i][j][k];
        return newArr;
    }

    private static void insertToMap(PhysicalGameState state, int[][][] map, int x, int y, String unitID, int player) {
        int[] oneHot = map[y][x];
        if(player==0) {
            oneHot[PLAYER0] = 1;
        }
        else {
            oneHot[PLAYER1] = 1;
        }
        switch(unitID){
            case "Resource": oneHot[RESOURCE]= 1; break;
            case "Light": oneHot[LIGHT]= 1; break;
            case "Worker": oneHot[WORKER]= 1; break;
            case "Heavy": oneHot[HEAVY]= 1;break;
            case "Base": oneHot[BASE]= 1;  break;
            case "Ranged": oneHot[RANGED]= 1;  break;
            case "Barracks": oneHot[BARRACKS]= 1;break;
            default:
                System.out.println(unitID);
        }
        oneHot[EMPTY]=0;
    }

    private static void CNN_INPUT(ArrayList<NeuralInput> inpArr) {
    }

    private static void initMap(int[][][] map) {
        for(int i=0;i<map.length;i++)
            for(int j=0;j<map[i].length;j++)
                for(int k=0;k<map[i][j].length;k++)
                    map[i][j][k]=0;
    }

    private static class NeuralInput {
        int[][][] arr;
        String p1;
        String p2;
        int idx;
    }
}
