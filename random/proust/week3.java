import edu.princeton.cs.algs4.StdRandom;

class merge {
    private static void merge(Comparable[] a, Comparable[] aux, int lo, int mid, int hi) {
        for (int k = lo; k <= hi; k++)
            aux[k] = a[k];
        int i = lo, j = mid + 1;
        for (int k = lo; k <= hi; k++) {
            if (i > mid) a[k] = aux[j++];
            else if (j > hi) a[k] = aux[i++];
            else if (less(aux[j], aux[i])) a[k] = aux[j++];
            else a[k] = aux[i++];
        }
    }

    private static boolean less(Comparable v, Comparable w) {
        return v.compareTo(w) < 0;
    }

    private static void sort(Comparable[] a, Comparable[] aux, int lo, int hi) {
        if (hi <= lo) return;
        int mid = lo + (hi - lo) / 2;
        sort(a, aux, lo, mid);
        sort(a, aux, mid + 1, hi);
        merge(a, aux, lo, mid, hi);
    }

    public static void sort(Comparable[] a) {
        Comparable[] aux = new Comparable[a.length];
        sort(a, aux, 0, a.length - 1);
    }
}

class Quick {
    public static void sort(Comparable[] a) {
        StdRandom.shuffle(a);
        sort(a, 0, a.length - 1);
    }

    private static void sort(Comparable[] a, int lo, int hi) {
        if (hi <= lo) return;
        int j = partition(a, lo, hi);
        sort(a, lo, j - 1);
        sort(a, j + 1, hi);
    }

    private static int partition(Comparable[] a, int lo, int hi) {
        int i = lo;
        int j = hi + 1;
        while (true) {
            while (less(a[++i], a[lo]))
                if (i == hi) break;
            while (less(a[lo], a[--j]))
                if (j == lo) break;
            if (i >= j) break;
            exch(a, i, j);
        }
        exch(a, lo, j);
        return j;
    }

    private static boolean less(Comparable v, Comparable w) {
        return v.compareTo(w) < 0;
    }

    private static void exch(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}


public class week3 {
    public static void main(String[] args) {
        int N = 10;
        int[] arr = new int[N];
        for (int i = 0; i < N / 2; i++) {
            arr[i] = i + 1;
            arr[i + N / 2] = i + 1;
        }

        for (Comparable i : arr) {
            System.out.print(i);
            System.out.print(" ");
        }
        System.out.print("\n");

        // merge
        int lo = 0;
        int mid = N / 2 - 1;
        int hi = N - 1;
        int[] ret = new int[N];
        int i = lo;
        int j = mid + 1;
        for (int k = 0; k < N; k++) {
            for (Comparable ii : ret) {
                System.out.print(ii);
                System.out.print(" ");
            }
            System.out.print("\n");

            System.out.print("i: ");
            System.out.print(i);
            System.out.print("\tj: ");
            System.out.print(j);

            System.out.print("\tarr[i]: ");
            System.out.print(arr[i]);
            System.out.print("\tarr[j]: ");
            System.out.print(arr[j]);
            System.out.print("\t<=: ");
            System.out.print(arr[i] <= arr[j]);
            System.out.print(" ");

            System.out.print("\n");

            if (j > hi) ret[k] = arr[i++]; // second array done, copy first
            else if (i > mid) ret[k] = arr[j++]; // first array done, copy second
            else if (arr[i] <= arr[j]) ret[k] = arr[i++]; // left is lower, copy it
            else ret[k] = arr[j++];
        }


        for (Comparable ii : ret) {
            System.out.print(ii);
            System.out.print(" ");
        }
        System.out.print("\n");


        // quack sort
        N = 10;
        for (int i = 0; i < N / 2; i++) {
            arr[i] = i + 1;
            arr[i + N / 2] = i + 1;
        }
        for (Comparable i : arr) {
            System.out.print(i);
            System.out.print(" ");
        }
        System.out.print("\n");


        int k = 0;
        i = 1;
        j = N - 1;
        while (i < j) {
            while (arr[i] < arr[k]) i++;
            while (arr[j] > arr[k]) j--;
            exch(arr, i, j);
        }


    }

    private static void exch(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}
