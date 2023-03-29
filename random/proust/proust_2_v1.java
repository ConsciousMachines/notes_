// ----------------------------------------------------------------------------
//                                pair
// ----------------------------------------------------------------------------

import java.util.HashMap;

class Pair<X, Y> {
    public final X x;
    public final Y y;

    public Pair(X x, Y y) {
        this.x = x;
        this.y = y;
    }
}

// ----------------------------------------------------------------------------
//                      parser for simple S-expressions
// ----------------------------------------------------------------------------

class Parser {
    // invariant: 'idx' always points to start of next lexeme, so when using
    // consume_char or consume_symbol, they eat up whitespace after.
    public int idx = 0;
    public String s = "";

    public char curr_char() {
        return s.charAt(idx);
    }

    public boolean char_is_whitespace(char c) {
        if ((c == ' ') || (c == '\n') || (c == '\t')) return true;
        return false;
    }

    public boolean char_is_reserved(char c) {
        if ((c == '(') || (c == ')')) return true;
        return false;
    }

    public void __consume_whitespace() {
        if (idx < s.length()) {
            while (char_is_whitespace(curr_char())) idx++;
        }
    }

    public String consume_symbol() {
        int start = idx;
        while (true) {
            char c = curr_char();
            if (char_is_reserved(c) || char_is_whitespace(c)) break;
            idx++;
        }
        String ret = s.substring(start, idx);
        __consume_whitespace();
        return ret;
    }

    public void consume_char(char c) {
        if (curr_char() == c) idx++;
        else throw new RuntimeException("expected '" + c + "'");
        __consume_whitespace();
    }

    public Type parse_type() {
        Type t = new Type();
        if (curr_char() == '(') {
            consume_char('(');
            Type t1 = parse_type();
            String type_op = consume_symbol();
            Type t2 = parse_type();
            consume_char(')');
            t.fst = t1;
            t.snd = t2;
            if (type_op.equals("->")) t.k = TypeKind.Arrow;
            else throw new RuntimeException("unknown operand in type-formula");
        }
        else {
            String symbol = consume_symbol();
            t.k = TypeKind.Var;
            t.symbol = symbol;
        }
        return t;
    }
}

// ----------------------------------------------------------------------------
//                      context is a linked list
// ----------------------------------------------------------------------------

class ContextNode {
    String symbol = null;
    Type type = null;
    ContextNode prev = null;

    public ContextNode() {
    }

    public ContextNode(String symbol, Type type) {
        this.symbol = symbol;
        this.type = type;
    }

    Type assoc(String symbol) {
        if (this.symbol.equals(symbol)) return this.type;
        if (this.prev == null) return null;
        return this.prev.assoc(symbol);
    }

    public String toString() {
        String ret = symbol + " : " + type + "\n";
        if (this.prev != null) ret += this.prev.toString();
        return ret;
    }
}

// ----------------------------------------------------------------------------
//                        nodes for building proof trees
// ----------------------------------------------------------------------------

enum TermType {Var, Lam, App, Ann, Hole}

class Term {
    TermType t;
    Term fst, snd;      // for compound types
    Type ann_type;      // for annotation
    String symbol;      // for var
    Integer hole_num;   // for hole, nullable

    public Term(TermType t, Term fst, Term snd, Type ann_type, String symbol, Integer hole_num) {
        this.t = t;
        this.fst = fst;
        this.snd = snd;
        this.ann_type = ann_type;
        this.symbol = symbol;
        this.hole_num = hole_num;
    }

    public String toString() {
        switch (this.t) {
            case Var:
                return this.symbol;
            case Lam:
                return "(\\" + fst + "." + snd + ")";
            case App:
                return "(" + fst + " " + snd + ")";
            case Ann:
                return "(" + fst + " : " + ann_type + ")";
            case Hole:
                return "?" + hole_num;
            default:
                throw new RuntimeException("unreachable");
        }
    }

    // static methods to easily create terms (for interactive proof)

    static Term var(String s) {
        return new Term(TermType.Var, null, null, null, s, 0);
    }

    static Term lam(String s, Term body) {
        return new Term(TermType.Lam, Term.var(s), body, null, null, 0);
    }

    static Term app(Term rator, Term rand) {
        return new Term(TermType.App, rator, rand, null, null, 0);
    }

    static Term ann(Term expr, Type ann_type) {
        return new Term(TermType.Ann, expr, null, ann_type, null, 0);
    }

    static Term hole(int hole_num) {
        return new Term(TermType.Hole, null, null, null, null, hole_num);
    }

    static Term hole() {
        return new Term(TermType.Hole, null, null, null, null, null);
    }
}

enum TypeKind {Var, Arrow}

class Type {
    TypeKind k;
    Type fst, snd; // for compound type
    String symbol; // for type variable

    public Type() {
    }

    public Type(TypeKind k, Type fst, Type snd, String symbol) {
        this.k = k;
        this.fst = fst;
        this.snd = snd;
        this.symbol = symbol;
    }

    public String toString() {
        switch (k) {
            case Arrow:
                return "(" + fst + " -> " + snd + ")";
            case Var:
                return symbol;
            default:
                throw new RuntimeException("unreachable");
        }
    }

    public boolean equals(Type t2) {
        if (this.k != t2.k) return false;
        switch (this.k) {
            case Var:
                return this.symbol.equals(t2.symbol);
            case Arrow:
                return this.fst.equals(t2.fst) && this.snd.equals(t2.snd);
            default:
                throw new RuntimeException("unreachable");
        }
    }
}

/*
public static class Parser {
    public static int start = 0;
    public static int idx = 0;
    public static String s = "";

    public static boolean char_is_whitespace(char c) {
        if ((c == ' ') || (c == '\n') || (c == '\t')) return true;
        return false;
    }

    public static boolean char_is_reserved(char c) {
        if ((c == '(') || (c == ')')) return true;
        return false;
    }

    public static void consume_whitespace() {
        while (char_is_whitespace(s.charAt(idx))) idx++;
        start = idx; // also synchronize start
    }

    public static String consume_symbol() {
        while (true) {
            char c = s.charAt(idx);
            if (char_is_reserved(c)) break;
            if (char_is_whitespace(c)) break;
            idx++;
        }
        String ret = s.substring(start, idx);
        start = idx; // synchronize start
        return ret;
    }

    public static char curr_char() {
        return s.charAt(idx);
    }

    public static void consume_char(char c) {
        if (s.charAt(idx) == c) {
            idx++;
            start = idx;
        }
        else throw new RuntimeException(
            "expected '" + c + "' but got something else");
    }

    public static void parse() {
        consume_char('(');
        while (true) {
            consume_whitespace();
            String symbol = consume_symbol();
            if (symbol.length() > 0) System.out.println(symbol);
            if (curr_char() == '(') parse();
            if (curr_char() == ')') break;
        }
        consume_char(')');
    }
}

*/


// ----------------------------------------------------------------------------
//                           the main proof assistant
// ----------------------------------------------------------------------------

class proust_2 {

    void type_check(ContextNode ctx, Term expr, Type type) {
        switch (expr.t) // Var, Lam, App, Ann, Hole
        {
            case Lam: // backwards of arrow introduction rule
                if (type.k == TypeKind.Arrow) {
                    // augment context with lam input
                    ContextNode ctx2 = new ContextNode(expr.fst.symbol, type.fst);
                    ctx2.prev = ctx;
                    // check that body has output type in new context
                    type_check(ctx2, expr.snd, type.snd);
                }
                else {
                    throw new RuntimeException("Lambda not matched with arrow type");
                }
                break;
            case Hole:
                if (refining) { // enter hole in goal table
                    goal_table.put(expr.hole_num, new Pair(type, ctx));
                }
                break;
            default:
                Type t0 = type_synth(ctx, expr);
                if (!type.equals(t0))
                    throw new RuntimeException("failed type synth:" + type + " vs " + t0);
        }
    }

    Type type_synth(ContextNode ctx, Term expr) {
        switch (expr.t) // Var, Lam, App, Ann, Hole
        {
            case Ann:
                type_check(ctx, expr.fst, expr.ann_type);
                return expr.ann_type;
            case App:
                Type tf = type_synth(ctx, expr.fst); // synthesize function's type
                if (tf.k == TypeKind.Arrow) {
                    type_check(ctx, expr.snd, tf.fst); // type check the input
                    return tf.snd;
                }
                else {
                    throw new RuntimeException("function type did not synthesize to Arrow");
                }
            case Var:
                Type t_from_ctx = ctx.assoc(expr.symbol);
                if (t_from_ctx == null) throw new RuntimeException("term not in context: " + expr);
                return t_from_ctx;
            default:
                throw new RuntimeException("failed to synthesize type");
        }
    }

    boolean refining = false;
    int hole_ctr;
    Parser parser;
    Term current_expr;
    HashMap<Integer, Pair<Type, ContextNode>> goal_table;

    int use_hole_ctr() {
        return hole_ctr++;
    }

    void set_task(String s) {
        parser = new Parser();
        parser.s = s;
        Type t = parser.parse_type();
        goal_table = new HashMap<>();
        hole_ctr = 1;
        current_expr = Term.ann(Term.hole(0), t);
        goal_table.put(0, new Pair(t, null));
        System.out.println("\nTask is now\n" + current_expr);
    }

    void print_goal() {
        for (int i : goal_table.keySet()) {
            System.out.println("Goal " + i + " has type: " + goal_table.get(i).x);
            System.out.println("in context\n" + goal_table.get(i).y);
        }
    }

    Term number_new_holes(Term e) {
        switch (e.t) {
            case Lam:
                return Term.lam(e.fst.symbol, number_new_holes(e.snd));
            case App:
                return Term.app(number_new_holes(e.fst), number_new_holes(e.snd));
            case Var:
                return e;
            case Ann:
                return Term.ann(number_new_holes(e.fst), e.ann_type);
            case Hole:
                if (e.hole_num != null) return Term.hole(e.hole_num);
                else return Term.hole(use_hole_ctr());
            default:
                throw new RuntimeException("unreachable in number_new_holes");
        }
    }

    Term replace_goal_with(int n, Term repl, Term e) {
        switch (e.t) {
            case Lam:
                return Term.lam(e.fst.symbol, replace_goal_with(n, repl, e.snd));
            case App:
                return Term.app(replace_goal_with(n, repl, e.fst),
                                replace_goal_with(n, repl, e.snd));
            case Var:
                return e;
            case Ann:
                return Term.ann(replace_goal_with(n, repl, e.fst), e.ann_type);
            case Hole:
                if (e.hole_num == n) return repl;
                else return Term.hole(e.hole_num);
            default:
                throw new RuntimeException("unreachable");
        }
    }

    void refine(int n, Term e) {
        Pair<Type, ContextNode> pair = goal_table.get(n);
        Type t = pair.x;
        ContextNode ctx = pair.y;
        type_check(ctx, e, t); // first time, just check
        Term en = number_new_holes(e);
        refining = true;
        type_check(ctx, en, t); // second time, add goals to table
        refining = false;
        goal_table.remove(n);
        current_expr = replace_goal_with(n, en, current_expr);
        System.out.println("\nTask with " + goal_table.size() + " goals is now\n" + current_expr);
        print_goal();
    }

    void test() {
        set_task("((B -> C) -> ((A -> B) -> (A -> C)))");
        refine(0, Term.lam("a", Term.hole()));
        refine(1, Term.lam("b", Term.hole()));
        refine(2, Term.lam("c", Term.hole()));
        refine(3, Term.app(Term.var("a"), Term.app(Term.var("b"), Term.var("c"))));
    }

    public static void main(String[] args) {
        proust_2 proust = new proust_2();
        proust.test();
    }
}


