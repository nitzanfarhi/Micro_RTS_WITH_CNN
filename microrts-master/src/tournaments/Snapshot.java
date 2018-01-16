package tournaments;

import rts.PhysicalGameState;

/**
 * Created by admin on 16/01/2018.
 */
public class Snapshot {
    int snapIDX;
    String p1;
    String p2;
    PhysicalGameState state;

    public Snapshot(int snapIDX, String p1, String p2, PhysicalGameState state) {
        this.snapIDX = snapIDX;
        this.p1 = p1;
        this.p2 = p2;
        this.state = state;
    }

}
