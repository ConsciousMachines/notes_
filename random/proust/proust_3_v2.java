import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

class VmapNode {
    String a;
    String b;
    VmapNode prev;

    public VmapNode(String a, String b, VmapNode prev) {
        this.a = a;
        this.b = b;
        this.prev = prev;
    }

    public String toString() {
        String ret = a + " : " + b + "\n";
        if (prev != null) ret += prev.toString();
        return ret;
    }
}

class CtxNode {
    String symbol;
    Expr expr;
    CtxNode prev;

    public CtxNode(String symbol, Expr expr, CtxNode prev) {
        this.symbol = symbol;
        this.expr = expr;
        this.prev = prev;
    }

    public String toString() {
        String ret = symbol + " : " + expr + "\n";
        if (prev != null) ret += prev.toString();
        return ret;
    }
}

enum ExprType {Var, Lam, App, Ann, Arrow, Type, Teq, Eq_refl, Eq_elim, True, False, Bool_ind, Bool}

class Expr {
    ExprType t;
    Expr e1, e2, e3, e4, e5;
    String s;

    private Expr(ExprType t, Expr e1, Expr e2, Expr e3, Expr e4, Expr e5, String s) {
        this.t = t;
        this.e1 = e1;
        this.e2 = e2;
        this.e3 = e3;
        this.e4 = e4;
        this.e5 = e5;
        this.s = s;
    }

    static Expr var(String name) {
        return new Expr(ExprType.Var, null, null, null, null, null, name);
    }

    static Expr lam(String var, Expr body) {
        return new Expr(ExprType.Lam, body, null, null, null, null, var);
    }

    static Expr app(Expr rator, Expr rand) {
        return new Expr(ExprType.App, rator, rand, null, null, null, null);
    }

    static Expr ann(Expr expr, Expr type) {
        return new Expr(ExprType.Ann, expr, type, null, null, null, null);
    }

    static Expr arrow(String var, Expr domain, Expr range) {
        return new Expr(ExprType.Arrow, domain, range, null, null, null, var);
    }

    static Expr type() {
        return new Expr(ExprType.Type, null, null, null, null, null, null);
    }

    static Expr teq(Expr left, Expr right) {
        return new Expr(ExprType.Teq, left, right, null, null, null, null);
    }

    static Expr eq_refl(Expr val) {
        return new Expr(ExprType.Eq_refl, val, null, null, null, null, null);
    }

    static Expr eq_elim(Expr x, Expr P, Expr px, Expr y, Expr peq) {
        return new Expr(ExprType.Eq_elim, x, P, px, y, peq, null);
    }

    static Expr true_() {
        return new Expr(ExprType.True, null, null, null, null, null, null);
    }

    static Expr false_() {
        return new Expr(ExprType.False, null, null, null, null, null, null);
    }

    static Expr bool_ind(Expr P, Expr tp, Expr fp, Expr y) {
        return new Expr(ExprType.Bool_ind, P, tp, fp, y, null, null);
    }

    static Expr bool() {
        return new Expr(ExprType.Bool, null, null, null, null, null, null);
    }

    public String toString() {
        switch (t) {
            case Var:
                return this.s;
            case Ann:
                return "(" + e1 + " : " + e2 + ")";
            case App:
                return "(" + e1 + " " + e2 + ")";
            case Teq:
                return "(" + e1 + " = " + e2 + ")";
            case Lam:
                return "(\\" + s + "." + e1 + ")";
            case Arrow:
                if (s.equals("_")) {
                    return "(" + e1 + " -> " + e2 + ")";
                }
                else {
                    return "(forall " + s + " : " + e1 + " -> " + e2 + ")";
                }
            case Bool_ind:
                return "(bool-ind " + e1 + " " + e2 + " " + e3 + " " + e4 + ")";
            case Eq_refl:
                return "(eq-refl " + e1 + ")";
            case Eq_elim:
                return "(eq-elim " + e1 + " " + e2 + " " + e3 + " " + e4 + " " + e5 + ")";
            case Bool:
                return "Bool";
            case True:
                return "true";
            case False:
                return "false";
            case Type:
                return "Type";
            default:
                throw new RuntimeException("unreachable");
        }
    }
}


class proust_2 {

    static Map<String, Expr> defs = new HashMap<>();
    static Map<String, Expr> deftypes = new HashMap<>();

    static void clear_defs() {
        defs = new HashMap<>();
        deftypes = new HashMap<>();
    }

    static void def(String name, Expr expr) {
        if (defs.containsKey(name)) {
            throw new RuntimeException(name + " already defined");
        }
        Expr e = expr; // no parsing
        Expr et = type_synth(null, e);
        switch (e.t) {
            case Ann: {
                Expr ex = e.e1;
                Expr tp = e.e2;
                defs.put(name, ex);
                deftypes.put(name, tp);
            }
            default: {
                defs.put(name, e);
                deftypes.put(name, et);
            }
        }
    }

    static void check_one(Expr expr) {
        System.out.println(type_synth(null, expr));
    }

    static void reduce_one(Expr expr) {
        System.out.println(reduce(null, expr));
    }

    static CtxNode assoc(String symbol, CtxNode ctx) {
        if (ctx.symbol.equals(symbol)) return ctx;
        if (ctx.prev == null) return null;
        return assoc(symbol, ctx.prev);
    }

    static VmapNode assoc(String symbol, VmapNode vmap) {
        if (vmap == null) return null;
        if (vmap.a.equals(symbol)) return vmap;
        if (vmap.prev == null) return null;
        return assoc(symbol, vmap.prev);
    }

    static Expr subst(String oldx, Expr newx, Expr expr) {
        switch (expr.t) {
            case Var: {
                String x = expr.s;
                return x.equals(oldx) ? newx : Expr.var(x);
            }
            case Arrow: {
                String x = expr.s;
                Expr t = expr.e1;
                Expr w = expr.e2;
                if (x.equals("_")) { // (Arrow '_ t w)
                    return Expr.arrow("_", subst(oldx, newx, t), subst(oldx, newx, w));
                }
                if (x.equals(oldx)) {
                    return expr;
                }
                if (side_cond(x, Arrays.asList(newx), false)) {
                    String repx = refresh(x, Arrays.asList(newx, w));
                    return Expr.arrow(repx, subst(oldx, newx, t),
                                      subst(oldx, newx, subst(x, Expr.var(repx), w)));
                }
                else {
                    return Expr.arrow(x, subst(oldx, newx, t), subst(oldx, newx, w));
                }
            }
            case Lam: {
                String x = expr.s;
                Expr b = expr.e1;
                if (x.equals("_")) { // (Lam '_ b)
                    return Expr.lam("_", subst(oldx, newx, b));
                }
                if (x.equals(oldx)) {
                    return expr;
                }
                if (side_cond(x, Arrays.asList(newx), false)) {
                    String repx = refresh(x, Arrays.asList(newx, b));
                    return Expr.lam(repx, subst(oldx, newx, subst(x, Expr.var(repx), b)));
                }
                else {
                    return Expr.lam(x, subst(oldx, newx, b));
                }
            }
            case App: {
                Expr f = expr.e1;
                Expr a = expr.e2;
                return Expr.app(subst(oldx, newx, f), subst(oldx, newx, a));
            }
            case Ann: {
                Expr e = expr.e1;
                Expr t = expr.e2;
                return Expr.ann(subst(oldx, newx, e), subst(oldx, newx, t));
            }
            case Type: {
                return Expr.type();
            }
            case Eq_refl: {
                Expr a = expr.e1;
                return Expr.eq_refl(subst(oldx, newx, a));
            }
            case Eq_elim: {
                Expr e1 = expr.e1;
                Expr e2 = expr.e2;
                Expr e3 = expr.e3;
                Expr e4 = expr.e4;
                Expr e5 = expr.e5;
                return Expr.eq_elim(subst(oldx, newx, e1), subst(oldx, newx, e2),
                                    subst(oldx, newx, e3), subst(oldx, newx, e4),
                                    subst(oldx, newx, e5));
            }
            case Teq: {
                Expr a = expr.e1;
                Expr b = expr.e2;
                return Expr.teq(subst(oldx, newx, a), subst(oldx, newx, b));
            }
            case True: {
                return Expr.true_();
            }
            case False: {
                return Expr.false_();
            }
            case Bool_ind: {
                Expr e1 = expr.e1;
                Expr e2 = expr.e2;
                Expr e3 = expr.e3;
                Expr e4 = expr.e4;
                return Expr.bool_ind(subst(oldx, newx, e1), subst(oldx, newx, e2),
                                     subst(oldx, newx, e3), subst(oldx, newx, e4));
            }
            case Bool: {
                return Expr.bool();
            }
            default: {
                throw new RuntimeException("unreachable");
            }
        }
    }

    static String freshen(String x) {
        return x + "_";
    }

    static String refresh(String x, List<Expr> lst) {
        if (side_cond(x, lst, true)) {
            return refresh(freshen(x), lst);
        }
        else {
            return x;
        }
    }

    static boolean side_cond(String x, List<Expr> lst, boolean check_binders) {
        for (Expr expr : lst) {
            if (sc_helper(x, expr, check_binders)) return true;
        }
        return false; // TODO: are these the semantics of ormap?
    }

    static boolean sc_helper(String x, Expr expr, boolean check_binders) {
        switch (expr.t) {
            case Var: {
                String y = expr.s;
                return x.equals(y);
            }
            case Arrow: {
                String y = expr.s;
                Expr tt = expr.e1;
                Expr tw = expr.e2;
                if (y.equals("_")) {
                    return side_cond(x, Arrays.asList(tt, tw), check_binders);
                }
                if (check_binders) {
                    return x.equals(y) || side_cond(x, Arrays.asList(tt, tw), check_binders);
                }
                else {
                    if (x.equals(y)) {
                        return false;
                    }
                    else {
                        return side_cond(x, Arrays.asList(tt, tw), check_binders);
                    }
                }
            }
            case Lam: {
                String y = expr.s;
                Expr b = expr.e1;
                if (y.equals("_")) { // (Lam '_ b)
                    return sc_helper(x, b, check_binders);
                }
                if (check_binders) {
                    return x.equals(y) || sc_helper(x, b, check_binders);
                }
                else {
                    if (x.equals(y)) {
                        return false;
                    }
                    else {
                        return sc_helper(x, b, check_binders);
                    }
                }
            }
            case App: {
                Expr f = expr.e1;
                Expr a = expr.e2;
                return side_cond(x, Arrays.asList(f, a), check_binders);
            }
            case Ann: {
                Expr e = expr.e1;
                Expr t = expr.e2;
                return side_cond(x, Arrays.asList(e, t), check_binders);
            }
            case Type: {
                return false;
            }
            case Eq_refl: {
                Expr a = expr.e1;
                return sc_helper(x, a, check_binders);
            }
            case Eq_elim: {
                Expr e1 = expr.e1;
                Expr e2 = expr.e2;
                Expr e3 = expr.e3;
                Expr e4 = expr.e4;
                Expr e5 = expr.e5;
                return side_cond(x, Arrays.asList(e1, e2, e3, e4, e5), check_binders);
            }
            case Teq: {
                Expr a = expr.e1;
                Expr b = expr.e2;
                return side_cond(x, Arrays.asList(a, b), check_binders);
            }
            case True: {
                return false;
            }
            case False: {
                return false;
            }
            case Bool_ind: {
                Expr e1 = expr.e1;
                Expr e2 = expr.e2;
                Expr e3 = expr.e3;
                Expr e4 = expr.e4;
                return side_cond(x, Arrays.asList(e1, e2, e3, e4), check_binders);
            }
            case Bool: {
                return false;
            }
            default: {
                throw new RuntimeException("unreachable");
            }
        }
    }

    static boolean used_in_ctx(CtxNode ctx, String x) {
        if (deftypes.containsKey(x)) return true;
        while (ctx != null) {
            if (side_cond(x, Arrays.asList(ctx.expr), false)) return true;
            ctx = ctx.prev;
        }
        return false; // TODO: ormap semantics?
    }

    static String refresh_with_ctx(CtxNode ctx, String x, List<Expr> lst) {
        if (side_cond(x, lst, true) || used_in_ctx(ctx, x)) {
            return refresh_with_ctx(ctx, freshen(x), lst);
        }
        else {
            return x;
        }
    }

    static boolean alpha_equiv(Expr e1, Expr e2) {
        return ae_helper(e1, e2, null);
    }

    static boolean ae_helper(Expr e1, Expr e2, VmapNode vmap) {
        if (e1.t != e2.t) return false; // types should match
        switch (e1.t) {
            case Var: {
                String x1 = e1.s;
                String x2 = e2.s;
                VmapNode xm1 = assoc(x1, vmap);
                return x2.equals(xm1 != null ? xm1.b : x1);
            }
            case Lam: {
                String x1 = e1.s;
                String x2 = e2.s;
                Expr b1 = e1.e1;
                Expr b2 = e2.e1;
                return ae_helper(b1, b2, new VmapNode(x1, x2, vmap));
            }
            case App: {
                Expr f1 = e1.e1;
                Expr a1 = e1.e2;
                Expr f2 = e2.e1;
                Expr a2 = e2.e2;
                return ae_helper(f1, f2, vmap) && ae_helper(a1, a2, vmap);
            }
            case Ann: {
                Expr _e1 = e1.e1;
                Expr t1 = e1.e2;
                Expr _e2 = e2.e1;
                Expr t2 = e2.e2;
                return ae_helper(_e1, _e2, vmap) && ae_helper(t1, t2, vmap);
            }
            case Arrow: {
                String x1 = e1.s;
                Expr t1 = e1.e1;
                Expr w1 = e1.e2;
                String x2 = e2.s;
                Expr t2 = e2.e1;
                Expr w2 = e2.e2;
                return ae_helper(t1, t2, new VmapNode(x1, x2, vmap)) &&
                        ae_helper(w1, w2, new VmapNode(x1, x2, vmap));
            }
            case Type: {
                return true;
            }
            case Eq_refl: {
                Expr x1 = e1.e1;
                Expr x2 = e2.e1;
                return ae_helper(x1, x2, vmap);
            }
            case Eq_elim: {
                Expr x1 = e1.e1;
                Expr P1 = e1.e2;
                Expr px1 = e1.e3;
                Expr y1 = e1.e4;
                Expr peq1 = e1.e5;
                Expr x2 = e2.e1;
                Expr P2 = e2.e2;
                Expr px2 = e2.e3;
                Expr y2 = e2.e4;
                Expr peq2 = e2.e5;
                return ae_helper(x1, x2, vmap) &&
                        ae_helper(P1, P2, vmap) &&
                        ae_helper(px1, px2, vmap) &&
                        ae_helper(y1, y2, vmap) &&
                        ae_helper(peq1, peq2, vmap);
            }
            case Teq: {
                Expr a1 = e1.e1;
                Expr b1 = e1.e2;
                Expr a2 = e2.e1;
                Expr b2 = e2.e2;
                return ae_helper(a1, a2, vmap) && ae_helper(b1, b2, vmap);
            }
            case True: {
                return true;
            }
            case False: {
                return true;
            }
            case Bool_ind: {
                Expr e11 = e1.e1;
                Expr e12 = e1.e2;
                Expr e13 = e1.e3;
                Expr e14 = e1.e4;
                Expr e21 = e2.e1;
                Expr e22 = e2.e2;
                Expr e23 = e2.e3;
                Expr e24 = e2.e4;
                return ae_helper(e11, e21, vmap) &&
                        ae_helper(e12, e22, vmap) &&
                        ae_helper(e13, e23, vmap) &&
                        ae_helper(e14, e24, vmap);
            }
            case Bool: {
                return true;
            }
            default: {
                throw new RuntimeException("unreachable");
            }
        }
    }

    static Expr reduce(CtxNode ctx, Expr expr) {
        switch (expr.t) {
            case Var: {
                String x = expr.s;
                {
                    CtxNode temp = assoc(x, ctx);
                    if (temp != null) {
                        return Expr.var(x);
                    }
                }
                if (defs.containsKey(x)) {
                    return reduce(ctx, defs.get(x));
                }
                else {
                    return Expr.var(x);
                }
            }
            case Arrow: {
                String x = expr.s;
                Expr a = expr.e1;
                Expr b = expr.e2;
                if (x.equals("_")) {
                    return Expr.arrow("_", reduce(ctx, a), reduce(ctx, b));
                }
                Expr ra = reduce(ctx, a);
                Expr rb = reduce(new CtxNode(x, ra, ctx), b);
                return Expr.arrow(x, ra, rb);
            }
            case Lam: {
                String x = expr.s;
                Expr b = expr.e1;
                if (x.equals("_")) {
                    return Expr.lam("_", reduce(ctx, b));
                }
                return Expr.lam(x, reduce(new CtxNode(x, null, ctx), b));
            }
            case App: {
                Expr f = expr.e1;
                Expr a = expr.e2;
                Expr fr = reduce(ctx, f);
                Expr fa = reduce(ctx, a);
                if (fr.t == ExprType.Lam) {
                    String x = fr.s;
                    Expr b = fr.e1;
                    return reduce(ctx, subst(x, fa, b));
                }
                else {
                    return Expr.app(fr, fa);
                }
            }
            case Ann: {
                Expr e = expr.e1;
                Expr t = expr.e2;
                return reduce(ctx, e);
            }
            case Type: {
                return Expr.type();
            }
            case Eq_refl: {
                Expr x = expr.e1;
                return Expr.eq_refl(reduce(ctx, x));
            }
            case Eq_elim: {
                Expr x = expr.e1;
                Expr P = expr.e2;
                Expr px = expr.e3;
                Expr y = expr.e4;
                Expr peq = expr.e5;
                Expr peqr = reduce(ctx, peq);
                if (peqr.t == ExprType.Eq_refl) {
                    return reduce(ctx, px);
                }
                else {
                    return Expr.eq_elim(reduce(ctx, x), reduce(ctx, P), reduce(ctx, px),
                                        reduce(ctx, y), peqr);
                }
            }
            case Teq: {
                Expr a = expr.e1;
                Expr b = expr.e2;
                return Expr.teq(reduce(ctx, a), reduce(ctx, b));
            }
            case True: {
                return Expr.true_();
            }
            case False: {
                return Expr.false_();
            }
            case Bool_ind: {
                Expr P = expr.e1;
                Expr tp = expr.e2;
                Expr fp = expr.e3;
                Expr b = expr.e4;
                Expr br = reduce(ctx, b);
                switch (br.t) {
                    case True:
                        return reduce(ctx, tp);
                    case False:
                        return reduce(ctx, fp);
                    default:
                        return Expr.bool_ind(reduce(ctx, P), reduce(ctx, tp), reduce(ctx, fp), br);
                }
            }
            case Bool: {
                return Expr.bool();
            }
            default: {
                throw new RuntimeException("unreachable");
            }
        }
    }

    static boolean equiv(CtxNode ctx, Expr e1, Expr e2) {
        return alpha_equiv(reduce(ctx, e1), reduce(ctx, e2));
    }

    static boolean type_check(CtxNode ctx, Expr expr, Expr type) {
        switch (expr.t) {
            case Lam: {
                String x = expr.s;
                Expr b = expr.e1;
                type_check(ctx, type, Expr.type());
                Expr tyr = reduce(ctx, type);
                switch (tyr.t) {
                    case Arrow: {
                        String x1 = tyr.s;
                        Expr tt = tyr.e1;
                        Expr tw = tyr.e2;
                        System.out.println("    x,x1: " + x + " " + x1 + " tyr: " + tyr);
                        if (x.equals("_") && x1.equals("_")) {
                            return type_check(ctx, b, tw);
                        }

                        if (x.equals("_")) {
                            if (!(side_cond(x1, Arrays.asList(b), false) || used_in_ctx(ctx, x1))) {
                                return type_check(new CtxNode(x1, tt, ctx), b, tw);
                            }
                            else {
                                String newx = refresh_with_ctx(ctx, x1, Arrays.asList(b, tyr));
                                return type_check(new CtxNode(newx, tt, ctx), b,
                                                  subst(x1, Expr.var(newx), tw));
                            }
                        }

                        if (x1.equals("_")) {
                            if (!(side_cond(x, Arrays.asList(tyr), false) || used_in_ctx(ctx, x))) {
                                return type_check(new CtxNode(x, tt, ctx), b, tw);
                            }
                            else {
                                String newx = refresh_with_ctx(ctx, x, Arrays.asList(b, tyr));
                                return type_check(new CtxNode(newx, tt, ctx),
                                                  subst(x, Expr.var(newx), b), tw);
                            }
                        }

                        else {
                            if (x.equals(x1) && (!used_in_ctx(ctx, x1))) {
                                return type_check(new CtxNode(x, tt, ctx), b, tw);
                            }
                            if (!(side_cond(x, Arrays.asList(tyr), true) || used_in_ctx(ctx, x))) {
                                return type_check(new CtxNode(x, tt, ctx), b,
                                                  subst(x1, Expr.var(x), tw));
                            }
                            else {
                                String newx = refresh_with_ctx(ctx, x, Arrays.asList(expr, tyr));
                                return type_check(new CtxNode(newx, tt, ctx),
                                                  subst(x, Expr.var(newx), b),
                                                  subst(x1, Expr.var(newx), tw));
                            }
                        }
                    }
                    default: {
                        throw new RuntimeException("\nfailed to typecheck:\n" + expr +
                                                           "\nfor type:\n" + type +
                                                           "\nin context:\n" + ctx);
                    }
                }
            }
            default: {
                if (equiv(ctx, type_synth(ctx, expr), type)) {
                    return true;
                }
                else {
                    throw new RuntimeException("\nfailed to typecheck:\n" + expr +
                                                       "\nfor type:\n" + type +
                                                       "\nin context:\n" + ctx);
                }
            }
        }
    }

    static Expr type_synth(CtxNode ctx, Expr expr) {
        switch (expr.t) {
            case Var: {
                String x = expr.s;
                {
                    CtxNode temp = assoc(x, ctx);
                    if (temp != null) {
                        return temp.expr;
                    }
                }
                if (deftypes.containsKey(x)) {
                    return deftypes.get(x);
                }
                else {
                    throw new RuntimeException("did not find \"" + x + "\" in ctx/deftypes");
                }
            }
            case Lam: {
                throw new RuntimeException("cannot synth Lam");
            }
            case App: {
                Expr f = expr.e1;
                Expr a = expr.e2;
                Expr t1 = reduce(ctx, type_synth(ctx, f));
                if (t1.t == ExprType.Arrow) {
                    String x = t1.s;
                    Expr tt = t1.e1;
                    Expr tw = t1.e2;
                    if (x.equals("_")) { // TODO: not sure what #when does in racket
                        if (type_check(ctx, a, tt)) {
                            return tw;
                        }
                        else {
                            throw new RuntimeException("not sure if this should be an error");
                        }
                    }
                    else {
                        if (type_check(ctx, a, tt)) {
                            return subst(x, a, tw);
                        }
                        else {
                            throw new RuntimeException("not sure if this should be an error");
                        }
                    }
                }
                else {
                    throw new RuntimeException("did not synth arrow type in app");
                }
            }
            case Ann: {
                Expr e = expr.e1;
                Expr t = expr.e2;
                type_check(ctx, t, Expr.type());
                type_check(ctx, e, t);
                return t;
            }
            case Arrow: {
                String x = expr.s;
                Expr tt = expr.e1;
                Expr tw = expr.e2;
                if (x.equals("_")) {
                    type_check(ctx, tt, Expr.type());
                    type_check(ctx, tw, Expr.type());
                    return Expr.type();
                }
                else {
                    type_check(ctx, tt, Expr.type());
                    // TODO (cons `(,x ,tt) ctx)
                    type_check(new CtxNode(x, tt, ctx), tw, Expr.type());
                    return Expr.type();
                }
            }
            case Type: {
                return Expr.type();
            }
            case Teq: {
                Expr e1 = expr.e1;
                Expr e2 = expr.e2;
                Expr t1 = type_synth(ctx, e1);
                type_check(ctx, e2, t1);
                return Expr.type();
            }
            case Eq_refl: {
                Expr x = expr.e1;
                type_synth(ctx, x);
                return Expr.teq(x, x);
            }
            case Eq_elim: {
                Expr x = expr.e1;
                Expr P = expr.e2;
                Expr px = expr.e3;
                Expr y = expr.e4;
                Expr peq = expr.e5;
                Expr A = type_synth(ctx, x);
                type_check(ctx, P, Expr.arrow("_", A, Expr.type()));
                Expr Pann = Expr.ann(P, Expr.arrow("_", A, Expr.type()));
                type_check(ctx, px, Expr.app(Pann, x));
                type_check(ctx, y, A);
                type_check(ctx, peq, Expr.teq(x, y));
                return Expr.app(Pann, y);
            }
            case True: {
                return Expr.bool();
            }
            case False: {
                return Expr.bool();
            }
            case Bool_ind: {
                Expr P = expr.e1;
                Expr tp = expr.e2;
                Expr fp = expr.e3;
                Expr b = expr.e4;
                type_check(ctx, P, Expr.arrow("_", Expr.bool(), Expr.type()));
                Expr Pann = Expr.ann(P, Expr.arrow("_", Expr.bool(), Expr.type()));
                type_check(ctx, tp, Expr.app(Pann, Expr.true_()));
                type_check(ctx, fp, Expr.app(Pann, Expr.false_()));
                type_check(ctx, b, Expr.bool());
                return Expr.app(Pann, b);
            }
            case Bool: {
                return Expr.type();
            }
            default: {
                throw new RuntimeException("unreachable");
            }
        }
    }

    static Expr var(String name) {
        return Expr.var(name);
    }

    static Expr lam(String var, Expr body) {
        return Expr.lam(var, body);
    }

    static Expr app(Expr rator, Expr rand) {
        return Expr.app(rator, rand);
    }

    static Expr ann(Expr expr, Expr type) {
        return Expr.ann(expr, type);
    }

    static Expr arrow(String var, Expr domain, Expr range) {
        return Expr.arrow(var, domain, range);
    }

    static Expr arrow(Expr domain, Expr range) {
        return Expr.arrow("_", domain, range);
    }

    static Expr type() {
        return Expr.type();
    }

    static Expr teq(Expr left, Expr right) {
        return Expr.teq(left, right);
    }

    static Expr eq_refl(Expr val) {
        return Expr.eq_refl(val);
    }

    static Expr eq_elim(Expr x, Expr P, Expr px, Expr y, Expr peq) {
        return Expr.eq_elim(x, P, px, y, peq);
    }

    static Expr true_() {
        return Expr.true_();
    }

    static Expr false_() {
        return Expr.false_();
    }

    static Expr bool_ind(Expr P, Expr tp, Expr fp, Expr y) {
        return Expr.bool_ind(P, tp, fp, y);
    }

    static Expr bool() {
        return Expr.bool();
    }

    public static void main(String[] args) {
        {
            Expr not_exp = lam("b", bool_ind(lam("b", bool()), true_(), false_(), var("b")));
            Expr not_type = arrow(bool(), bool());
            Expr not = ann(not_exp, not_type);
            def("not", not);
            System.out.println(not);
        }
        {
            Expr and_expr = lam("b1", lam("b2", bool_ind(lam("b", bool()), var("b2"), false_(),
                                                         var("b1"))));
            Expr and_type = arrow(bool(), arrow(bool(), bool()));
            Expr and = ann(and_expr, and_type);
            def("and", and);
            System.out.println(and);
        }
        {
            Expr P_expr = lam("b", teq(var("b"), app(app(var("and"), var("b")), var("b"))));
            Expr P_type = arrow(bool(), type());
            Expr P = ann(P_expr, P_type);
            def("P", P);
            System.out.println(P);
        }
        {
            Expr ex11_expr = lam("b", bool_ind(var("P"), eq_refl(true_()),
                                               eq_refl(false_()), var("b")));
            Expr ex11_type = arrow("b", bool(), app(var("P"), var("b")));
            Expr ex11 = ann(ex11_expr, ex11_type);
            def("ex11", ex11);
            System.out.println(ex11);
        }
    }
}

// what makes dependent types do what they do:
// Equality brings terms into type space
// functions from terms to types P are the properties of the terms.
// terms go into types in Forall
// ???
//

// so Teq brings terms into type land. for ex11 we can evaluate the
// expression b = ((and b) b) for both cases in bool-ind to prove that
// this holds for all bools. so far this is 'brute force'. i would like
// to see where the logical part comes in (induction on nats?)

// so inside type-synth, bool-ind checks that the type returned by (P true)
// equals (true = true); and same for false. then it outputs that (P b)
// holds in general.

/*


((\b.(bool-ind P (eq-refl true) (eq-refl false) b)) : (forall b : Bool -> (P b)))
this goes into type-synth (not check! from def()) into ann case

check RHS is type
check LHS has this type.

	(\b.(bool-ind P (eq-refl true) (eq-refl false) b))
		: (forall b : Bool -> (P b))
	we are checking that this lambda has the type, in type-check, lam case
	check the type is type.
	tyr = reduce-type,
		(forall b : Bool -> (b = (bool-ind (\b_.Bool) b false b)))
	it is an arrow, as it should be.
	add b: Bool to ctx
	check lambda body has type (b = (bool-ind (\b_.Bool) b false b))

		(bool-ind P (eq-refl true) (eq-refl false) b)
			: (b = (bool-ind (\b_.Bool) b false b))
		checking. goes into synth into bool-ind case

			check (refl t) has type (P t)
			(refl t) goes into synth to come out as (t = t)
			it is then equiv-ed against (P t)
			they get reduced and alpha-compared

			check (refl f) has type (P f)

			return (P b)

		when it comes out of synth, (P b) is checked for equiv against
			(b = (bool-ind (\b_.Bool) b false b))
		in equiv, they are both reduced and alpha-equived.
 */
