class quick_find {
    public int[] id;
    private int n;

    public quick_find(int n) {
        this.n = n;
        id = new int[n];
        for (int i = 0; i < n; i++) id[i] = i;
    }

    public boolean find(int p, int q) {
        return id[p] == id[q];
    }

    public void union(int p, int q) {
        int pid = id[p];
        int qid = id[q];
        for (int i = 0; i < this.n; i++) if (id[i] == qid) id[i] = pid;
    }
}

class quick_union {
    public int[] id;
    private int n;

    public quick_union(int n) {
        this.n = n;
        id = new int[n];
        for (int i = 0; i < n; i++) id[i] = i;
    }

    private int get_root(int p) {
        while (id[p] != p) p = id[p];
        return p;
    }

    public void union(int p, int q) {
        int rp = get_root(p);
        int rq = get_root(q);
        id[rp] = rq;
    }

    public boolean find(int p, int q) {
        return get_root(p) == get_root(q);
    }
}

class quick_union_weight {
    public int[] id;
    public int[] sz;
    private int n;

    public quick_union_weight(int n) {
        this.n = n;
        this.id = new int[n];
        this.sz = new int[n];
        for (int i = 0; i < n; i++) {
            id[i] = i;
            sz[i] = 1;
        }
    }

    private int get_root(int p) {
        while (id[p] != p) p = id[p];
        return p;
    }

    public void union(int p, int q) {
        int rp = get_root(p);
        int rq = get_root(q);
        if (sz[rp] >= sz[rq]) {
            id[rq] = rp;
            sz[rp] = Math.max(sz[rp], sz[rq] + 1);
            //sz[rp] += sz[rq];
            // ok sedgewick says sz[rp] += sz[rq] which counts all elements
            // but my solution only counts the height of the tree
            // https://stackoverflow.com/questions/30957644/why-does-the-weighted-quick-union-algorithm-consider-the-sizes-of-the-tree-inste
            // also my height solution doesn't play well with path compression
            // and size is related to height as n <=> 2^n
            // so with path compression size is a better usage
            // the last comment showed they have the same upper bound
            // also for that weird example, adding the bigger & shorter tree
            // to the smaller and taller tree would make a lot more long paths
            // to find the root. but adding the small tall tree to the big
            // short tree creates fewer long paths. (randomness of merging)
        }
        else {
            id[rp] = rq;
            sz[rq] = Math.max(sz[rq], sz[rp] + 1);
            //sz[rq] += sz[rp];
        }
    }

    public boolean find(int p, int q) {
        return get_root(p) == get_root(q);
    }
}

class quick_union_comp {
    public int[] id;
    public int[] sz;
    private int n;

    public quick_union_comp(int n) {
        this.n = n;
        this.id = new int[n];
        this.sz = new int[n];
        for (int i = 0; i < n; i++) {
            id[i] = i;
            sz[i] = 1;
        }
    }

    private int get_root(int p) {
        int start = p; // save start for compression loop
        while (p != id[p]) {
            //id[p] = id[id[p]]; // compression 2
            p = id[p];
        }
        // now iterate again and do compression
        while (start != id[start]) { // same iteration
            int curr = start; // save current node
            start = id[start]; // jump to next node as usual
            id[curr] = p; // set the root of the node just visited
        }
        return p;
    }

    public void union(int p, int q) {
        int rp = get_root(p);
        int rq = get_root(q);
        if (sz[rp] >= sz[rq]) {
            id[rq] = rp;
            sz[rp] += sz[rq];
        }
        else {
            id[rp] = rq;
            sz[rq] += sz[rp];
        }
    }

    public boolean find(int p, int q) {
        return get_root(p) == get_root(q);
    }
}

public class week1 {
    public static void main(String[] args) {

        //quick_find qf = new quick_find(10);
        //quick_union qf = new quick_union(10);
        //quick_union_weight qf = new quick_union_weight(10);
        quick_union_comp qf = new quick_union_comp(10);

        qf.union(4, 3);
        qf.union(3, 8);
        qf.union(6, 5);
        qf.union(9, 4);
        qf.union(2, 1);
        qf.union(5, 0);
        qf.union(2, 7);
        qf.union(6, 1);
        qf.union(7, 3);

        System.out.print("    i: ");
        for (int i = 0; i < 10; i++) {
            System.out.print(i);
            System.out.print(" ");
        }
        System.out.print("\nid[i]: ");

        for (int i = 0; i < 10; i++) {
            System.out.print(qf.id[i]);
            System.out.print(" ");
        }
        System.out.print("\nsz[i]: ");

        for (int i = 0; i < 10; i++) {
            System.out.print(qf.sz[i]);
            System.out.print(" ");
        }
        System.out.print("\n");
    }
}
