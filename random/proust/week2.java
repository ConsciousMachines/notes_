import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

class Nod {
    public int value;
    public Nod next;
}

class Node<T> {
    public T value;
    public Node<T> prev;
    //public Node<T> next;

    public Node(T value) {
        this.value = value;
    }
}

class Stack<T> implements Iterable<T> {
    public int height = 0;
    private Node<T> tail;

    public Iterator<T> iterator() {
        return new ListIterator();
    }

    private class ListIterator implements Iterator<T> {
        private Node<T> current = tail;

        public boolean hasNext() {
            return current != null;
        }

        public T next() {
            T v = current.value;
            current = current.prev;
            return v;
        }
    }

    public Stack() {
        this.tail = new Node<T>(null);
    }

    public void push(T value) {
        Node<T> n = new Node<T>(value);
        //this.tail.next = n;
        n.prev = this.tail;
        this.tail = n;
        this.height++;
    }

    public T pop() {
        if (this.height == 0) throw new RuntimeException("empty stack");
        T ret = this.tail.value;
        this.tail = this.tail.prev;
        this.height--;
        return ret;
    }

    public boolean isEmpty() {
        return this.height == 0;
    }

    public String toString() {
        List<T> list = new ArrayList<T>();
        Node<T> ptr = this.tail;
        for (int i = 0; i < this.height; i++) // traverse list backwards
        {
            list.add(ptr.value);
            ptr = ptr.prev;
        }
        Collections.reverse(list);
        StringBuilder ret = new StringBuilder();
        ret.append("stack: ");
        for (T i : list) {
            ret.append(i);
            ret.append(" ");
        }
        return ret.toString();
    }
}

class ArrayStack<T> implements Iterable<T> {
    int size = 2;
    T[] data;
    private int ptr_next = 0;

    public ArrayStack() {
        data = (T[]) new Object[size];
    }

    public Iterator<T> iterator() {
        return new ReverseArrayIterator();
    }

    private class ReverseArrayIterator implements Iterator<T> {
        private int i = 0;

        public boolean hasNext() {
            return i < ptr_next;
        }

        public T next() {
            i++;
            return data[i - 1];
        }
    }

    private void resize_array(int new_size) {
        T[] new_data = (T[]) new Object[new_size];
        for (int i = 0; i < ptr_next; i++) new_data[i] = data[i];
        size = new_size;
        data = new_data;
    }

    public void push(T v) {
        data[ptr_next] = v;
        ptr_next++;
        if (ptr_next > (size - 1)) resize_array(size * 2);
    }

    public T pop() {
        if (ptr_next == 0) throw new RuntimeException("empty stack");
        ptr_next--;
        if ((ptr_next < (size / 4)) && (size > 3)) resize_array(size / 2);
        T ret = data[ptr_next];
        //data[ptr_next] = null; // remove reference for GC
        return ret; // the last item on the stack
    }

    public String toString() {
        StringBuilder ret = new StringBuilder();
        ret.append("array_stack: ");
        for (T i : this) {
            ret.append(i);
            ret.append(" ");
        }
        for (int i = ptr_next; i < size; i++) {
            ret.append("_ ");
        }
        return ret.toString();
    }
}

class Queue {
    Nod earliest = null;
    Nod latest = null;
    int length = 0;

    public void enqueue(int v) {
        Nod n = new Nod();
        n.value = v;
        if (length == 0) earliest = n;
        else latest.next = n;
        latest = n;
        length++;
    }

    public int dequeue() {
        if (length == 0) throw new RuntimeException("empty queue");
        int ret = earliest.value;
        earliest = earliest.next;
        length--;
        //if (length == 0) latest = null; // not necessary
        return ret;
    }

    public String toString() {
        StringBuilder ret = new StringBuilder();
        ret.append("queue: ");
        ret.append(length);
        ret.append(":\t");
        Nod ptr = earliest;
        for (int i = 0; i < length; i++) {
            ret.append(ptr.value);
            ret.append(" ");
            ptr = ptr.next;
        }
        return ret.toString();
    }
}

class Selection {
    public static void sort(Comparable[] a) {
        int N = a.length;
        for (int i = 0; i < N - 1; i++) {
            int idx_smolest = i;
            for (int j = i + 1; j < N; j++) {
                if (less(a[j], a[idx_smolest]))
                    idx_smolest = j;
            }
            exch(a, i, idx_smolest);
        }
    }

    private static boolean less(Comparable v, Comparable w) {
        return v.compareTo(w) < 0;
    }

    private static void exch(Comparable[] a, int i, int j) {
        Comparable swap = a[i];
        a[i] = a[j];
        a[j] = swap;
    }
}

class Insertion {
    public static void sort(Comparable[] a) {
        int N = a.length;
        for (int i = 1; i < N; i++) {
            int curr = i;
            int prev = i - 1;
            while (less(a[curr], a[prev])) {
                exch(a, curr, prev);
                curr--;
                prev--;
                if (prev == -1) break;
            }
        }
    }

    private static boolean less(Comparable v, Comparable w) {
        return v.compareTo(w) < 0;
    }

    private static void exch(Comparable[] a, int i, int j) {
        Comparable swap = a[i];
        a[i] = a[j];
        a[j] = swap;
        for (Comparable x : a) {
            System.out.print(x);
            System.out.print(" ");
        }
        System.out.print("\n");
    }
}

class ShellSort {
    // idea: elements move one element at a time
    // even tho they have long way to go.
    // big increment = small subarray
    // lets make the array partially sorted (which insertion is good at)
    // i think we are on average making it partially ordered. the
    // only way it wouldn't work is
    // if all similar magnitude numbers occur at h steps
    // otherwise we get most of the small numbers to the left, large to right,
    // and now it's partially ordered.
    // a g-sorted array is still g-sorted after h-sorting it
    public static void sort(Comparable[] a) {
        int N = a.length;
        int h = 1;
        while (h < N / 3) h = 3 * h + 1;
        while (h >= 1) {
            for (int i = h; i < N; i++) {
                for (int j = i; j >= h && less(a[j], a[j - h]); j -= h) {
                    exch(a, j, j - h);
                }
            }
            h = h / 3;
        }
    }

    private static boolean less(Comparable v, Comparable w) {
        return v.compareTo(w) < 0;
    }

    private static void exch(Comparable[] a, int i, int j) {
        Comparable swap = a[i];
        a[i] = a[j];
        a[j] = swap;
        for (Comparable x : a) {
            System.out.print(x);
            System.out.print(" ");
        }
        System.out.print("\n");
    }
}

public class week2 {
    public static void main(String[] args) {

        //Stack<Integer> s = new Stack<Integer>();
        ArrayStack<Integer> s = new ArrayStack<Integer>();

        for (int i = 0; i < 10; i++) {
            s.push(i);
            System.out.println(s);
        }
        for (int i = 0; i < 10; i++) {
            s.pop();
            System.out.println(s);
        }

        /*
        Stack<String> s2 = new Stack<String>();
        s2.push("dad");
        s2.push("soy");
        System.out.println(s2);
        s2.pop();
        System.out.println(s2);
        */

        Queue q = new Queue();

        int N = 5;
        for (int i = 0; i < N; i++) {
            q.enqueue(i);
            System.out.println(q);
        }
        for (int i = 0; i < N; i++) {
            q.dequeue();
            System.out.println(q);
        }

        // SORTING
        N = 10;
        Comparable[] arr = new Comparable[N];
        for (int i = 0; i < N; i++) {
            arr[i] = N - i - 1;
        }
        for (Comparable i : arr) {
            System.out.print(i);
            System.out.print(" ");
        }
        System.out.print("\n");
        //Selection sort = new Selection();
        //Insertion sort = new Insertion();
        ShellSort sort = new ShellSort();
        sort.sort(arr);

    }
}
