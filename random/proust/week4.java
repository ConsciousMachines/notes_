/* *****************************************************************************
 *  Name:              Ada Lovelace
 *  Coursera User ID:  123456
 *  Last modified:     October 16, 1842
 **************************************************************************** */

class Heap {
    public void sort(Comparable[] pq) {
        int N = pq.length;
        for (int k = N / 2; k >= 1; k--)
            sink(pq, k, N);
        while (N > 1) {
            exch(pq, 1, N);
            sink(pq, 1, --N);
        }

    }

    private boolean less(Comparable[] pq, int i, int j) {
        return pq[i].compareTo(pq[j]) < 0;
    }

    private void exch(Comparable[] pq, int i, int j) {
        Comparable t = pq[i];
        pq[i] = pq[j];
        pq[j] = t;
    }

    void sink(Comparable[] pq, int k, int N) {
        while (2 * k <= N) // while childs exist
        {
            int j = 2 * k; // left child
            if (j < N && less(pq, j, j + 1)) j++; // right child if its bigger
            if (!less(pq, k, j)) break; // if root greater than child, break (heap order is fine)
            exch(pq, k, j); // exchange child & root
            k = j; // new position of "root"
        }
    }
}

class MaxPQ<Key extends Comparable<Key>> {
    private Key[] pq;
    private int N;

    public MaxPQ(int capacity) {
        pq = (Key[]) new Comparable[capacity + 1];
    }

    public boolean isEmpty() {
        return N == 0;
    }

    private boolean less(int i, int j) {
        return pq[i].compareTo(pq[j]) < 0;
    }

    private void exch(int i, int j) {
        Key t = pq[i];
        pq[i] = pq[j];
        pq[j] = t;
    }

    void swim(int k) {
        while (k > 1 && less(k / 2, k)) {
            exch(k, k / 2);
            k = k / 2; // parent of node at k is k/2
        }

    }

    void insert(Key x) {
        pq[++N] = x;
        swim(N);
    }

    void sink(int k) {
        while (2 * k <= N) // while childs exist
        {
            int j = 2 * k; // left child
            if (j < N && less(j, j + 1)) j++; // right child if its bigger
            if (!less(k, j)) break; // if root greater than child, break (heap order is fine)
            exch(k, j); // exchange child & root
            k = j; // new position of "root"
        }
    }

    Key delMax() {
        Key max = pq[1];
        exch(1, N--);
        sink(1);
        pq[N + 1] = null;
        return max;
    }
}


public class week4 {
    public static void main(String[] args) {


    }
}
