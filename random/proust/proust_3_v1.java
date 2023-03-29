// ----------------------------------------------------------------------------
//                                pair
// ----------------------------------------------------------------------------

import java.util.Arrays;
import java.util.List;

class ListNode<X, Y> {
    public final X x;
    public final Y y;
    ListNode<X, Y> prev;

    public ListNode(X x, Y y, ListNode<X, Y> prev) {
        this.x = x;
        this.y = y;
        this.prev = prev;
    }

    ListNode<X, Y> assoc(X x) {
        if (this.x.equals(x)) return this;
        if (this.prev == null) return null;
        return this.prev.assoc(x);
    }
}

// ----------------------------------------------------------------------------
//                      context is a linked list
// ----------------------------------------------------------------------------

class CtxNode {
    String symbol;
    Expr expr;
    CtxNode prev;

    public CtxNode(String symbol, Expr expr, CtxNode prev) {
        this.symbol = symbol;
        this.expr = expr;
        this.prev = prev;
    }

    static CtxNode assoc(String symbol, CtxNode ctx) {
        if (ctx.symbol.equals(symbol)) return ctx;
        if (ctx.prev == null) return null;
        return assoc(symbol, ctx.prev);
    }

    public String toString() {
        String ret = symbol + " : " + expr + "\n";
        if (prev != null) ret += prev.toString();
        return ret;
    }
}

// ----------------------------------------------------------------------------
//                              Expression
// ----------------------------------------------------------------------------

enum ExprType {Var, Lam, App, Ann, Arrow, Type}

class Expr {
    final ExprType type;
    final Expr fst, snd;
    final String string;

    static CtxNode deftypes = null;
    static CtxNode defs = null;

    static void def(String name, Expr expr) {
        if (CtxNode.assoc(name, defs) != null) {
            throw new RuntimeException(name + " already defined");
        }
        Expr e = expr;
        Expr et = e.type_synth(null);
        switch (e.type) {
            case Ann:
                defs = new CtxNode(name, e.fst, defs);
                deftypes = new CtxNode(name, e.snd, deftypes);
                break;
            default:
                defs = new CtxNode(name, e, defs);
                deftypes = new CtxNode(name, et, deftypes);
                break;
        }
    }

    static void clear_defs() {
        defs = null;
        deftypes = null;
    }

    static void check_one(Expr expr) {
        System.out.println(expr.type_synth(null));
    }

    static void reduce_one(Expr expr) {
        System.out.println(expr.reduce(null));
    }

    static boolean side_cond(String x, List<Expr> lst, boolean check_binders) {
        for (Expr e : lst) {
            if (sc_helper(x, e, check_binders)) return true;
        }
        return false; // TODO: is this correct in terms of 'ormap'?
    }

    static boolean sc_helper(String x, Expr expr, boolean check_binders) {
        switch (expr.type) {
            case Var:
                return x.equals(expr.string);
            case Arrow:
                if (expr.string.equals("_")) { // (Arrow '_ tt tw)
                    return side_cond(x, Arrays.asList(expr.fst, expr.snd), check_binders);
                }
                if (check_binders) { // (Arrow y tt tw)
                    return x.equals(expr.string) || side_cond(x, Arrays.asList(expr.fst, expr.snd),
                                                              check_binders);
                }
                if (x.equals(expr.string)) {
                    return false;
                }
                return side_cond(x, Arrays.asList(expr.fst, expr.snd), check_binders);
            case Lam:
                if (expr.string.equals("_")) { // (Lam '_ b)
                    return sc_helper(x, expr.fst, check_binders);
                }
                if (check_binders) { // (Lam y b)
                    return x.equals(expr.string) || sc_helper(x, expr.fst, check_binders);
                }
                if (x.equals(expr.string)) {
                    return false;
                }
                return sc_helper(x, expr.fst, check_binders);
            case App:
                return side_cond(x, Arrays.asList(expr.fst, expr.snd), check_binders);
            case Ann:
                return side_cond(x, Arrays.asList(expr.fst, expr.snd), check_binders);
            case Type:
                return false;
            default:
                throw new RuntimeException("unreachable");
        }
    }

    static String freshen(String x) {
        return x + "_";
    }

    static String refresh(String x, List<Expr> lst) {
        if (side_cond(x, lst, true)) {
            return refresh(freshen(x), lst);
        }
        return x;
    }

    static boolean used_in_ctx(CtxNode ctx, String x) {
        if (CtxNode.assoc(x, Expr.deftypes) != null) return true;
        while (ctx != null) {
            if (side_cond(x, Arrays.asList(ctx.expr), false)) return true;
            ctx = ctx.prev;
        }
        return false;
    }

    static String refresh_with_ctx(CtxNode ctx, String x, List<Expr> lst) {
        if (side_cond(x, lst, true) || used_in_ctx(ctx, x)) {
            return refresh_with_ctx(ctx, freshen(x), lst);
        }
        else return x;
    }

    static boolean ae_helper(Expr fst, Expr snd, ListNode<String, String> vmap) {
        if (fst.type != snd.type) return false;
        ListNode<String, String> new_cell = new ListNode<>(fst.string, snd.string, vmap);
        switch (fst.type) {
            case Var:
                ListNode<String, String> xm1 = null;
                if (vmap != null) xm1 = vmap.assoc(fst.string);
                return snd.string.equals(xm1 != null ? xm1.y : fst.string);
            case Lam:
                return ae_helper(fst.fst, snd.fst, new_cell);
            case App:
            case Ann:
                return ae_helper(fst.fst, snd.fst, vmap) && ae_helper(fst.snd, snd.snd, vmap);
            case Arrow:
                return ae_helper(fst.fst, snd.fst, new_cell) && ae_helper(fst.snd, snd.snd,
                                                                          new_cell);
            case Type:
                return true;
            default:
                throw new RuntimeException("unreachable");
        }
    }

    static boolean alpha_equiv(Expr fst, Expr snd) {
        return ae_helper(fst, snd, null);
    }

    static boolean equiv(CtxNode ctx, Expr fst, Expr snd) {
        return alpha_equiv(fst.reduce(ctx), snd.reduce(ctx));
    }

    static Expr var(String s) {
        return new Expr(ExprType.Var, null, null, s);
    }

    static Expr lam(String s, Expr e) {
        return new Expr(ExprType.Lam, e, null, s);
    }

    static Expr app(Expr fst, Expr snd) {
        return new Expr(ExprType.App, fst, snd, null);
    }

    static Expr ann(Expr e, Expr t) {
        return new Expr(ExprType.Ann, e, t, null);
    }

    static Expr arrow(String x, Expr fst, Expr snd) {
        return new Expr(ExprType.Arrow, fst, snd, x);
    }

    static Expr type() {
        return new Expr(ExprType.Type, null, null, null);
    }

    private Expr(ExprType type, Expr fst, Expr snd, String string) {
        this.type = type;
        this.fst = fst;
        this.snd = snd;
        this.string = string;
    }

    Expr subst(String oldx, Expr newx) {
        switch (type) {
            case Var:
                if (string.equals(oldx)) return newx;
                else return this;
            case Arrow:
                if (string.equals("_")) {
                    return Expr.arrow("_", fst.subst(oldx, newx), snd.subst(oldx, newx));
                }
                if (this.string.equals(oldx)) {
                    return this;
                }
                if (side_cond(this.string, Arrays.asList(newx), false)) {
                    String repx = refresh(this.string, Arrays.asList(newx, this.snd));
                    return Expr.arrow(repx, fst.subst(oldx, newx),
                                      snd.subst(string, Expr.var(repx)).subst(oldx, newx));
                }
                else {
                    return Expr.arrow(string, fst.subst(oldx, newx), snd.subst(oldx, newx));
                }
            case Lam:
                if (string.equals("_")) {
                    return Expr.lam("_", fst.subst(oldx, newx));
                }
                if (this.string.equals(oldx)) {
                    return this;
                }
                if (side_cond(string, Arrays.asList(newx), false)) {
                    String repx = refresh(string, Arrays.asList(newx, fst));
                    return Expr.lam(repx, fst.subst(string, Expr.var(repx)).subst(oldx, newx));
                }
                else {
                    return Expr.lam(string, fst.subst(oldx, newx));
                }
            case App:
                return Expr.app(fst.subst(oldx, newx), snd.subst(oldx, newx));
            case Ann:
                return Expr.ann(fst.subst(oldx, newx), snd.subst(oldx, newx));
            case Type:
                return this;
            default:
                throw new RuntimeException("unreachable");
        }
    }

    public String toString() {
        switch (type) {
            case Var:
                return string;
            case Lam:
                return "(\\" + string + " => " + fst + ")";
            case App:
                return "(" + fst + " " + snd + ")";
            case Ann:
                return "(" + fst + " : " + snd + ")";
            case Arrow:
                if (string == "_") {
                    return "(" + fst + " -> " + snd + ")";
                }
                return "(forall (" + string + " : " + fst + ") -> " + snd + ")";
            case Type:
                return "Type";
            default:
                throw new RuntimeException("unreachable");
        }
    }

    Expr reduce(CtxNode ctx) {
        switch (type) {
            case Var:
                if (CtxNode.assoc(string, ctx) != null) return this;
                CtxNode tmp = CtxNode.assoc(string, defs);
                if (tmp != null) {
                    return tmp.expr.reduce(ctx);
                }
                else return this;
            case Arrow:
                if (string.equals("_")) {
                    return Expr.arrow("_", fst.reduce(ctx), snd.reduce(ctx));
                }
                Expr ra = fst.reduce(ctx);
                CtxNode neew = new CtxNode(string, ra, ctx);
                Expr rb = snd.reduce(neew);
                return Expr.arrow(string, ra, rb);
            case Lam:
                if (string.equals("_")) {
                    return Expr.lam("_", fst.reduce(ctx));
                }
                CtxNode new_node = new CtxNode(string, null, ctx);
                return Expr.lam(string, fst.reduce(new_node));
            case App:
                Expr fr = fst.reduce(ctx);
                Expr fa = snd.reduce(ctx);
                if (fr.type == ExprType.Lam) {
                    Expr ret = fr.fst.subst(fr.string, fa);
                    return ret.reduce(ctx);
                }
                else return Expr.app(fr, fa);
            case Ann:
                return fst.reduce(ctx);
            case Type:
                return this;
            default:
                throw new RuntimeException("unreachable");
        }
    }

    boolean type_check(CtxNode ctx, Expr type) {
        CtxNode ctx2;
        switch (this.type) {
            case Lam:
                String x = this.string;
                Expr b = this.fst;
                type.type_check(ctx, Expr.type());
                Expr tyr = type.reduce(ctx);
                switch (tyr.type) {
                    case Arrow:
                        String x1 = tyr.string;
                        Expr tt = tyr.fst;
                        Expr tw = tyr.snd;
                        if (x.equals("_") && x1.equals("_")) {
                            return b.type_check(ctx, tw);
                        }
                        if (x.equals("_")) {
                            if (!(side_cond(x1, Arrays.asList(b), false) || used_in_ctx(ctx, x1))) {
                                return b.type_check(new CtxNode(x1, tt, ctx), tw);
                            }
                            else {
                                String newx = refresh_with_ctx(ctx, x1, Arrays.asList(b, tyr));
                                ctx2 = new CtxNode(newx, tt, ctx);
                                return b.type_check(ctx2, tw.subst(x1, Expr.var(newx)));
                            }
                        }
                        if (x1.equals("_")) {
                            if (!(side_cond(x, Arrays.asList(tyr), false) || used_in_ctx(ctx, x))) {
                                return b.type_check(new CtxNode(x, tt, ctx), tw);
                            }
                            else {
                                String newx = refresh_with_ctx(ctx, x, Arrays.asList(b, tyr));
                                ctx2 = new CtxNode(newx, tt, ctx);
                                return b.subst(x, Expr.var(newx)).type_check(ctx2, tw);
                            }
                        }
                        else {
                            if (x.equals(x1) && !(used_in_ctx(ctx, x1))) {
                                ctx2 = new CtxNode(x, tt, ctx);
                                return b.type_check(ctx2, tw);
                            }
                            if (!(side_cond(x, Arrays.asList(tyr), true) || used_in_ctx(ctx, x))) {
                                ctx2 = new CtxNode(x, tt, ctx);
                                return b.type_check(ctx2, tw.subst(x1, Expr.var(x)));
                            }
                            else {
                                String newx = refresh_with_ctx(ctx, x, Arrays.asList(this, tyr));
                                ctx2 = new CtxNode(newx, tt, ctx);
                                return b.subst(x, Expr.var(newx))
                                        .type_check(ctx2, tw.subst(x1, Expr.var(newx)));
                            }
                        }
                    default:
                        throw new RuntimeException("reduced type is not arrow");
                }
            default:
                if (equiv(ctx, this.type_synth(ctx), type)) {
                    return true;
                }
                else throw new RuntimeException("failed to check: " + type + " in ctx:\n" + ctx);
        }
    }

    Expr type_synth(CtxNode ctx) {
        switch (type) {
            case Var:
                CtxNode res;
                if ((res = CtxNode.assoc(string, ctx)) != null) return res.expr;
                if ((res = CtxNode.assoc(string, deftypes)) != null) return res.expr;
                else throw new RuntimeException("didnt find type in ctx or deftypes.");
            case Lam:
                throw new RuntimeException("cant synth lam");
            case App:
                Expr t1 = fst.type_synth(ctx).reduce(ctx);
                if (t1.type == ExprType.Arrow) {
                    if (t1.string.equals("_")) {
                        if (snd.type_check(ctx, t1.fst)) {
                            return t1.snd;
                        }
                    }
                    else {
                        if (snd.type_check(ctx, t1.fst)) {
                            return t1.snd.subst(t1.string, snd);
                        }
                    }
                    throw new RuntimeException("couldn't synthesize APP");
                }
                else throw new RuntimeException("app fn was not an arrow");
            case Ann:
                snd.type_check(ctx, Expr.type());
                fst.type_check(ctx, snd);
                return snd;
            case Arrow:
                if (string.equals("_")) {
                    fst.type_check(ctx, Expr.type());
                    snd.type_check(ctx, Expr.type());
                    return Expr.type();
                }
                else {
                    fst.type_check(ctx, Expr.type());
                    CtxNode neew = new CtxNode(string, fst, ctx);
                    snd.type_check(neew, Expr.type());
                    return Expr.type();
                }
            case Type:
                return this;
            default:
                throw new RuntimeException("unreachable");
        }
    }
}

class proust_2 {
    public static void main(String[] args) {


    }
}


