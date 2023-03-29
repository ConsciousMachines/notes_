#lang racket
; file contains Unicode characters - download rather than cut/paste from browser

(provide def check-one reduce-one clear-defs parse-expr reduce type-check type-synth equiv?)

;; proust-pred-forall: minimal predicate logic
;; Prabhakar Ragde (August 2020)
;; with contributions from Astra Kolomatskaia

;; Grammar:

;; expr = (λ x => expr)             ; lambda abstraction
;;      | (expr expr)               ; function application
;;      | x                         ; variable
;;      | (expr : expr)             ; expression annotated with type
;;      | (∀ (x : expr) -> expr)    ; dependent function type
;;      | (expr -> expr)            ; special case equiv to (∀ (_ : expr) -> expr)
;;      | Type                      ; the type of types


;; Structures for Expr (abstract syntax tree representing terms).

(struct Lam (var body) #:transparent)
(struct App (rator rand) #:transparent)
(struct Ann (expr type) #:transparent)
(struct Arrow (var domain range) #:transparent)
(struct Type () #:transparent)


;; A Context is a (Listof (List Symbol Expr))
;; association list mapping variables to expressions.

;; Parsing

;; parse-expr : sexp -> Expr

(define (parse-expr s)
  (match s
    [`(λ ,(? symbol? x) => ,e) (Lam x (parse-expr e))]
    [`(∀ (,(? symbol? x) : ,t) -> ,e) (Arrow x (parse-expr t) (parse-expr e))]
    [`(∀ (,(? symbol? x) : ,t) ,(? list? a) ... -> ,e) (Arrow x (parse-expr t) (parse-expr `(∀ ,@a -> ,e)))]
    [`(,t1 -> ,t2) (Arrow '_ (parse-expr t1) (parse-expr t2))]
    [`(,t1 -> ,t2 -> ,r ...) (Arrow '_ (parse-expr t1) (parse-expr `(,t2 -> ,@r)))]
    [`(,e : ,t) (Ann (parse-expr e) (parse-expr t))]
    ['Type (Type)]
    [`(,e1 ,e2) (App (parse-expr e1) (parse-expr e2))]
    [`(,e1 ,e2 ,e3 ,r ...) (parse-expr `((,e1 ,e2) ,e3 ,@r))]
    ['_ (error 'parse "cannot use underscore\n")]
    [(? symbol? x) x]
    [else (error 'parse "bad syntax: ~a\n" s)]))

;; Unparsing, that is, pretty-printing.

;; pretty-print-expr : Expr -> String

(define (pretty-print-expr e)
  (match e
    [(Lam x b) (format "(λ ~a => ~a)" x (pretty-print-expr b))]
    [(App e1 e2) (format "(~a ~a)" (pretty-print-expr e1) (pretty-print-expr e2))]
    [(? symbol? x) (symbol->string x)]
    [(Ann e t) (format "(~a : ~a)" (pretty-print-expr e) (pretty-print-expr t))]
    [(Arrow '_ t1 t2) (format "(~a -> ~a)" (pretty-print-expr t1) (pretty-print-expr t2))]
    [(Arrow x t1 t2) (format "(∀ (~a : ~a) -> ~a)" x (pretty-print-expr t1) (pretty-print-expr t2))]
    [(Type) "Type"]))

;; pretty-print-context : Context -> String

(define (pretty-print-context ctx)
  (cond
    [(empty? ctx) ""]
    [else (string-append (format "\n~a : ~a" (first (first ctx)) (pretty-print-expr (second (first ctx))))
                         (pretty-print-context (rest ctx)))]))

;; Substitution
;; subst : Var Expr Expr -> Expr

(define (subst oldx newx expr)
  (match expr
    [(? symbol? x) (if (equal? x oldx) newx x)]
    [(Arrow '_ t w) (Arrow '_ (subst oldx newx t) (subst oldx newx w))]
    [(Arrow x t w)
       (cond
         [(equal? x oldx) expr]
         [(side-cond? x (list newx) false)
            (define repx (refresh x (list newx w)))
            (Arrow repx (subst oldx newx t) (subst oldx newx (subst x repx w)))]
         [else (Arrow x (subst oldx newx t) (subst oldx newx w))])]
    [(Lam '_ b) (Lam '_ (subst oldx newx b))]
    [(Lam x b)
       (cond
         [(equal? x oldx) expr]
         [(side-cond? x (list newx) false)
            (define repx (refresh x (list newx b)))
            (Lam repx (subst oldx newx (subst x repx b)))]
         [else (Lam x (subst oldx newx b))])]
    [(App f a) (App (subst oldx newx f) (subst oldx newx a))]
    [(Ann e t) (Ann (subst oldx newx e) (subst oldx newx t))]
    [(Type) (Type)]))

;; Helpers for substitution

;; refresh : Var (Listof Expr) -> Var

(define (refresh x lst)
  (if (side-cond? x lst true) (refresh (freshen x) lst) x))

(define (freshen x) (string->symbol (string-append (symbol->string x) "_")))

;; checks if variable x occurs free in lst (list of expressions)
;; and, when check-binders? is #t, if x has a binding occurrence in lst

(define (side-cond? x lst check-binders?)
  (ormap (lambda (expr) (sc-helper x expr check-binders?)) lst))

(define (sc-helper x expr check-binders?)
  (match expr
    [(? symbol? y) (equal? x y)]
    [(Arrow '_ tt tw) (side-cond? x (list tt tw) check-binders?)]
    [(Arrow y tt tw)
       (cond
         [check-binders? (or (equal? x y) (side-cond? x (list tt tw) check-binders?))]
         [else (if (equal? x y) false (side-cond? x (list tt tw) check-binders?))])]
    [(Lam '_ b) (sc-helper x b check-binders?)]
    [(Lam y b)
       (cond
         [check-binders? (or (equal? x y) (sc-helper x b check-binders?))]
         [else (if (equal? x y) false (sc-helper x b check-binders?))])]
    [(App f a) (side-cond? x (list f a) check-binders?)]
    [(Ann e t) (side-cond? x (list e t) check-binders?)]
    [(Type) false]))

;; Code for finding fresh variable name not already used in a context
;; (similar to refresh, above) to be used later in type checking/synthesis

(define (used-in-ctx? ctx x)
  (or (assoc x deftypes) (ormap (λ (p) (side-cond? x (rest p) false)) ctx)))

(define (refresh-with-ctx ctx x lst)
  (if (or (side-cond? x lst true) (used-in-ctx? ctx x)) (refresh-with-ctx ctx (freshen x) lst) x))

;; Alpha equivalence (produces #t/#f)

(define (alpha-equiv? e1 e2) (ae-helper e1 e2 empty))

;; vmap is association list mapping variables in e1 to variables in e2

(define (ae-helper e1 e2 vmap)
  (match (list e1 e2)
    [(list (? symbol? x1) (? symbol? x2))
       (define xm1 (assoc x1 vmap))
       (equal? (if xm1 (second xm1) x1) x2)]
    [(list (Lam x1 b1) (Lam x2 b2)) (ae-helper b1 b2 (cons (list x1 x2) vmap))]
    [(list (App f1 a1) (App f2 a2)) (and (ae-helper f1 f2 vmap) (ae-helper a1 a2 vmap))]
    [(list (Ann e1 t1) (Ann e2 t2)) (and (ae-helper e1 e2 vmap) (ae-helper t1 t2 vmap))]
    [(list (Arrow x1 t1 w1) (Arrow x2 t2 w2))
       (and (ae-helper t1 t2 (cons (list x1 x2) vmap)) (ae-helper w1 w2 (cons (list x1 x2) vmap)))]
    [(list (Type) (Type)) true]
    [else false]))

;; Reduction

;; reduce: Context Expr -> Expr
;; Reduces an expression by recursively looking up definitions
;; and doing substitution for applications

(define (reduce ctx expr)
  (match expr
    [(? symbol? x)
        (cond
         [(assoc x ctx) x]
         [(assoc x defs) => (lambda (p) (reduce ctx (second p)))]
         [else x])]
    [(Arrow '_ a b)
       (Arrow '_ (reduce ctx a) (reduce ctx b))]
    [(Arrow x a b)
       (define ra (reduce ctx a))
       (define rb (reduce (cons (list x ra) ctx) b))
       (Arrow x ra rb)]
    [(Lam '_ b) (Lam '_ (reduce ctx b))]
    [(Lam x b) (Lam x (reduce (cons (list x '()) ctx) b))]
    [(App f a)
       (define fr (reduce ctx f))
       (define fa (reduce ctx a))
       (match fr
         [(Lam x b) (reduce ctx (subst x fa b))]
         [else (App fr fa)])]
    [(Ann e t) (reduce ctx e)]
    [(Type) (Type)]))

;; Definitional equality

(define (equiv? ctx e1 e2) (alpha-equiv? (reduce ctx e1) (reduce ctx e2))) 

;; This is the heart of the verifier.
;; type-check and type-synth are mutually-recursive functions that
;;   check an expression has a type in a context, and
;;   synthesize the type of an expression in a context, respectively.
;; They implement bidirectional type checking.

;; type-check : Context Expr Expr -> boolean
;; Produces true if expr has type t in context ctx (or error if not).

(define (type-check ctx expr type)
  (match expr
    [(Lam x b)
       (type-check ctx type (Type))
       (define tyr (reduce ctx type))
       (match tyr
         [(Arrow x1 tt tw)
            (match (list x x1)
              [(list '_ '_) (type-check ctx b tw)]
              [(list '_ x1)
                 (cond
                   [(nor (side-cond? x1 (list b) false) (used-in-ctx? ctx x1))
                      (type-check (cons (list x1 tt) ctx) b tw)]
                   [else
                      (define newx (refresh-with-ctx ctx x1 (list b tyr)))
                      (type-check (cons (list newx tt) ctx) b (subst x1 newx tw))])]
              [(list x '_)
                 (cond
                   [(nor (side-cond? x (list tyr) false) (used-in-ctx? ctx x))
                      (type-check (cons (list x tt) ctx) b tw)]
                   [else
                      (define newx (refresh-with-ctx ctx x (list b tyr)))
                      (type-check (cons (list newx tt) ctx) (subst x newx b) tw)])]
              [(list x x1)
                 (cond
                   [(and (equal? x x1) (not (used-in-ctx? ctx x1))) (type-check (cons (list x tt) ctx) b tw)]
                   [(nor (side-cond? x (list tyr) true) (used-in-ctx? ctx x))
                      (type-check (cons (list x tt) ctx) b (subst x1 x tw))]
                   [else
                      (define newx (refresh-with-ctx ctx x (list expr tyr)))
                      (type-check (cons (list newx tt) ctx) (subst x newx b) (subst x1 newx tw))])])]
         [else (cannot-check ctx expr type)])]
    [else (if (equiv? ctx (type-synth ctx expr) type) true (cannot-check ctx expr type))]))

;; cannot-check: Context Expr Type -> void
;; Prints generic error message for type-check
;; Error messages can be improved by replacing uses of this with something more specific

(define (cannot-check ctx e t)
  (error 'type-check "cannot typecheck ~a as ~a in context:\n~a"
         (pretty-print-expr e) (pretty-print-expr t) (pretty-print-context ctx)))

;; type-synth : Context Expr -> Type
;; Produces type of expr in context ctx (or error if can't)

(define (type-synth ctx expr)
  (match expr
    [(? symbol? x)
       (cond
         [(assoc x ctx) => second]
         [(assoc x deftypes) => second]
         [else (cannot-synth ctx expr)])]
    [(Lam x b) (cannot-synth ctx expr)]
    [(App f a) 
       (define t1 (reduce ctx (type-synth ctx f)))
       (match t1
         [(Arrow '_ tt tw) #:when (type-check ctx a tt) tw]
         [(Arrow x tt tw) #:when (type-check ctx a tt) (subst x a tw)]
         [else (cannot-synth ctx expr)])]
    [(Ann e t) (type-check ctx t (Type)) (type-check ctx e t) t]
    [(Arrow '_ tt tw)
       (type-check ctx tt (Type))
       (type-check ctx tw (Type))
       (Type)]
    [(Arrow x tt tw)
       (type-check ctx tt (Type))
       (type-check (cons `(,x ,tt) ctx) tw (Type))
       (Type)]
    [(Type) (Type)]))

;; cannot-synth: Context Expr -> void
;; Prints generic error message for type-synth
;; Again, can improve things by being more specific at use sites

(define (cannot-synth ctx expr)
  (error 'type-synth "cannot infer type of ~a in context:\n~a"
         (pretty-print-expr expr) (pretty-print-context ctx)))

;; Definitions

;; Global variables

(define defs empty)
(define deftypes empty)

;; def : symbol Expr -> void
;; Side effect is to mutate global defs, deftypes to include new def if it checks out

(define (def name expr)
  (when (assoc name defs) (error 'def "~a already defined" name))
  (define e (parse-expr expr))
  (define et (type-synth empty e))
  (match e
    [(Ann ex tp) (set! defs (cons (list name ex) defs))
                 (set! deftypes (cons (list name tp) deftypes))]
    [else (set! defs (cons (list name e) defs))
          (set! deftypes (cons (list name et) deftypes))]))

(define (clear-defs) (set! defs empty) (set! deftypes empty))

(define (check-one expr)
  (printf "~a\n" (pretty-print-expr (type-synth empty (parse-expr expr)))))

(define (reduce-one expr)
   (printf "~a\n" (pretty-print-expr (reduce empty (parse-expr expr)))))


; ============================================================================
; ============================================================================


; bool - this type describes a function that takes a type, then 2 inputs of
; that type, and returns an output of that type. since this holds for any type
; it must be returning one of the 2 arguments. so there can be 2 such funcs
(def 'bool '((∀ (X : Type) -> (X -> X -> X)) : Type))

; true - takes a type, ignores it, takes 2 things and returns the first
; it behaves according to the specification by the bool type. same for false
(def 'true '((λ x => (λ y => (λ z => y))) : bool))
(def 'false '((λ x => (λ y => (λ z => z))) : bool))

; if - takes a type, a bool, and two inputs of that type. feeds type to the
; bool to 'instantiate' it polymorphically, then applies the 2 inputs to it
; if bool is t then we get the first thing, if f the second.
(def 'if '((λ X => (λ b => (λ t => (λ f => (b X t f)))))
           : (∀ (X : Type) -> (bool -> (X -> (X -> X))))))

; band - takes 2 bools, returns a bool. if x is true, returns y. if x is
; false, returns false.
(def 'band '((λ x => (λ y => (x bool y false)))
             : (bool -> (bool -> bool))))

; and - takes 2 types and returns a type (note the thing starting with
; forall is a type term, parameterized by p and q)
; so a lambda whose body term is a type.

; what's the difference between having (Type -> Type) in the type, vs
; having (∀ (c : Type) -> Type) ??? Nothing. in the first one, the
; binding is thrown out so we never reuse that type.
; we could replace "∀ (c : Type)" with "Type" but then we can't require
; that this first type is present again in the statement as c.
; c's presence puts constraints on our inputs and outputs (describes them)

; the returned type specifies: consume type c, consume (function that
; consumes p, q, returns c) and return c.
; if c is BOB_SAGET then this is impossible. we only have p,q to work with,
; thus c must be constructed out of p,q 
; thus c must be p or q. we apply and to p q p, p q q to get another type:
; it consumes a function of type p q p or p q q and returns p/q respectively
; these are the functions that take 2 things, return first / second

; TLDR: "and" consumes types p q to yield a type which requires
; consuming p or q, then consuming \x.\y.x or \x.\y.y to yield p or q.
(def 'and '((λ p => (λ q => (∀ (c : Type) -> ((p -> (q -> c)) -> c))))
             : (Type -> (Type -> Type))))

; conj - consume type P,Q and elements a,b of type P,Q.
; want to return type ((and p) q), which from above we know is a type that
; consumes c, consumes first/second fn, and returns type c.
; NOTE: used polymorphism using c to include both first/second functions.
; NOTE: we know f : (p -> (q -> p/q)) thus must be applied to a then b

; we want AND to contain 2 things (somewhere). these 2 things must be extracted
; using first/second. since we are in a typed setting, we need to use polymorphism
; for AND to accept both first/second. thus AND accepts p + first or q + second,
; then respectively returns p/q. think of it as: in order to type check,
; AND consumes a tuple <c, fn> which comes out as <p, first> or <q, second>

; NOTE: the type inputs of the proof term are ignored for now because we only
; yield terms, not other types (so far). the type info is needed for type check

; * * * AND takes 2 types, and produces a thing that is ready for elimination!!!
; ie it is ready to accept <p, first> or <q, second> in order to get eliminated.
; since it accepts eliminators (and necessary polymorphic type variables),
; those eliminators make up the type of the encoding!!!
; so a general encoding should be (∀ (c : Type) -> (elim -> c))
; where c is any of its output types.

; so where are the p,q stored? they are stored in ((((conj P) Q) p) q) which is
; actually just the lambda term \C.\f.((f a) b) meaning we ignore C and accept
; a first/second eliminator to get back the element p/q of type P/Q.
; so we are expecting an eliminator (so write lambda to wait for it to be input)
; and then apply it to the stored information. the info is captured in a CLOSURE
(def 'conj '((λ p => (λ q => (λ a => (λ b => (λ C => (λ f => ((f a) b)))))))
             : (∀ (p : Type) -> (∀ (q : Type) -> (p -> (q -> ((and p) q)))))))

; proj - consume types P,Q, then consume ((and p) q) and return p
; the type of ((and p) q) is a function that takes c,first/second and returns c
; so we first take in P,Q,f
; we want f to return p. so it must consume <p,first>

; figure it out the long way:
;(def 'proj1 '((λ p => (λ q => (λ f => ((f p) (λ p => (λ q => p))))))
;              : (∀ (p : Type) -> (∀ (q : Type) -> ((∀ (c : Type) -> ((p -> (q -> c)) -> c)) -> p)))))
;(def 'proj2 '((λ p => (λ q => (λ f => ((f q) (λ p => (λ q => q))))))
;              : (∀ (p : Type) -> (∀ (q : Type) -> ((∀ (c : Type) -> ((p -> (q -> c)) -> c)) -> q)))))

(def 'proj1 '((λ p => (λ q => (λ f => ((f p) (λ x => (λ y => x))))))
              : (∀ (p : Type) -> (∀ (q : Type) -> (((and p) q) -> p)))))

(def 'proj2 '((λ p => (λ q => (λ f => ((f q) (λ x => (λ y => y))))))
              : (∀ (p : Type) -> (∀ (q : Type) -> (((and p) q) -> q)))))

; and-commutes - consume types P,Q, consume ((and p) q) and return ((and q) p)
; so probably want to deconstruct these things and reassemble them
; - we want to create a thing of type ((and q) p) in the end so use ((conj q) p) <- note reversed types
; - then for the projections its easy. apply them in reverse order to conj
(def 'and-commutes '((λ p => (λ q => (λ a => ((((conj q) p) (((proj2 p) q) a)) (((proj1 p) q) a)))))
                     : (∀ (p : Type) -> (∀ (q : Type) -> (((and p) q) -> ((and q) p))))))

; nat
; RAGDE: we never actually have any instances of type X... why?
; RAGDE: i'd think we use nested pairs rather than id for succ...

; TODO: i dont really get whats happening with nat

; nat - consume type x
; consume f : ((x -> x) -> x)
; return x
(def 'nat '((∀ (x : Type) -> (((x -> x) -> x) -> x))
            : Type))

; zero - consume type X,
; consume f : (X -> X) -> X
; return X
; f takes in (X -> X). we don't have any such thing except identity.
; applying f to identity returns X as needed.
(def 'z '((λ X => (λ f => (f (λ a => a))))
          : nat))

; succ - consumes nat. to return a nat we need to create one somehow.
; start outputting nat's form: consume type X,
; then consume f : ((X -> X) -> X)
; all we have to work with is n, X, f, nat (not z because it's not general)
; (n nat) seems too recursive
; (n X) has type (((X -> X) -> X) -> X)
; so it consumes f : ((X -> X) -> X) and returns X
(def 's '((λ n => (λ X => (λ f => ((n X) f))))
          : (nat -> nat)))

; when we apply (s z) we get:
; (\n.\X.\f.((n X) f) \X.\f.(f \a.a))
; \X.\f.( ( \X.\f.(f \a.a) X) f)      now apply X to the z term
; \X.\f.( \f.(f \a.a) f)              now apply f to the z term
; \X.\f.(f \a.a) 
(def 'one '((s z) : nat))
; (s (s z))
; (s \X.\f.(f \a.a))
; (\n.\X.\f.((n X) f) \X.\f.(f \a.a)) apply (s z) as input n to s
; \X.\f.((\X.\f.(f \a.a) X) f)        apply X to the inner thing
; \X.\f.(\f.(f \a.a) f)               apply f
; \X.\f.(f \a.a)                      we get the same thing as before - why?
; assume correct since he mentions we cant prove 1 =/= 0
; RAGDE: is this correct or is there a limitation to this typed nat
(def 'two '((s (s z)) : nat))

; plus - consume two nats, start outputting nat's form
; consume type X
; consume function f : ((X -> X) -> X)
; all we have to work with is: n1, n2, X, f, s,
; a nat consumes <X,f> to make a closure where f is applied to previous structure.
; n1 : nat
; n2 : nat
; X : Type
; f : (X -> X) -> X
; s : nat -> nat
(def 'plus '((λ n1 => (λ n2 => (λ X => (λ f => n1))))
             : (nat -> (nat -> nat))))



;(equiv? '() 'one 'two)




; in untyped LC, to create a pair, we take 2 items and output a closure
; that awaits its eliminator, the first/second function.
; the interpretation is that we use it correctly by passing in first/second as f.
; \x.\y.\f.((f x) y)

; the typed version of this requires a type for everything. we had 3 items before
; so this can be 3 additional type arguments:
; \X.\Y. will be type inputs for x,y
; f must be X->Y->{X/Y} so it needs a type input.


; TODO: logical OR, NOT, lists 

; TODO: why is one == two?
; (equiv? '() 'one 'two)

;(def 'plus '(? : (nat -> (nat -> nat))))

;(def 'second '((λ X => (λ a => (λ b => b)))
;               : (∀ (X : Type) -> (X -> (X -> X)))))

;(def 'id '((λ X => (λ x => x))
;           :(∀ (X : Type) -> (X -> X))))





