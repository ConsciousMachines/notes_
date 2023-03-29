#lang racket
; file contains Unicode characters - download rather than cut/paste from browser

(provide def check-one reduce-one clear-defs parse-expr reduce type-check type-synth)

;; proust-pred-eq: for-all, equality.
;; Prabhakar Ragde (September 2021)
;; with contributions from Astra Kolomatskaia

;; Grammar:

;; expr = (λ x => expr)             ; lambda abstraction
;;      | (expr expr)               ; function application
;;      | x                         ; variable
;;      | (expr : expr)             ; expression annotated with type
;;      | (∀ (x : expr) -> expr)    ; dependent function type
;;      | (expr -> expr)            ; special case equiv to (∀ (_ : expr) -> expr)
;;      | Type                      ; the type of types
;;      | (expr = expr)             ; equality types
;;      | (eq-refl x)               ; equality introduction
;;      | (eq-elim expr expr expr expr expr) ; equality elimination


;; Structures for Expr (abstract syntax tree representing terms).

(struct Lam (var body) #:transparent)
(struct App (rator rand) #:transparent)
(struct Ann (expr type) #:transparent)
(struct Arrow (var domain range) #:transparent)
(struct Type () #:transparent)
(struct Teq (left right) #:transparent)
(struct Eq-refl (val) #:transparent)
(struct Eq-elim (x P px y peq) #:transparent)
(struct Z () #:transparent)
(struct S (pred) #:transparent)
(struct Nat () #:transparent)
(struct Nat-ind (P zc sc n) #:transparent)

;; A Context is a (Listof (List Symbol Expr))
;; association list mapping variables to expressions.

;; Parsing

;; parse-expr : sexp -> Expr

(define (parse-expr sss)
  (match sss
    [`(λ ,(? symbol? x) => ,e) (Lam x (parse-expr e))]
    [`(∀ (,(? symbol? x) : ,t) -> ,e) (Arrow x (parse-expr t) (parse-expr e))]
    [`(∀ (,(? symbol? x) : ,t) ,(? list? a) ... -> ,e) (Arrow x (parse-expr t) (parse-expr `(∀ ,@a -> ,e)))]
    [`(,t1 -> ,t2) (Arrow '_ (parse-expr t1) (parse-expr t2))]
    [`(,t1 -> ,t2 -> ,r ...) (Arrow '_ (parse-expr t1) (parse-expr `(,t2 -> ,@r)))]
    [`(,e : ,t) (Ann (parse-expr e) (parse-expr t))]
    ['Type (Type)]
    ['Nat (Nat)]
    [`(nat-ind ,e1 ,e2 ,e3 ,e4)
        (Nat-ind (parse-expr e1) (parse-expr e2) (parse-expr e3) (parse-expr e4))]
    ['z (Z)]
    [`(s ,e) (S (parse-expr e))]
    [`(eq-refl ,a) (Eq-refl (parse-expr a))]
    [`(eq-elim ,e1 ,e2 ,e3 ,e4 ,e5)
        (Eq-elim (parse-expr e1) (parse-expr e2)  (parse-expr e3)  (parse-expr e4)  (parse-expr e5))]
    [`(,e1 = ,e2) (Teq (parse-expr e1) (parse-expr e2))]
    [`(,e1 ,e2) (App (parse-expr e1) (parse-expr e2))]
    [`(,e1 ,e2 ,e3 ,r ...) (parse-expr `((,e1 ,e2) ,e3 ,@r))]
    ['_ (error 'parse "cannot use underscore\n")]
    [(? symbol? x) x]
    [else (error 'parse "bad syntax: ~a\n" sss)]))

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
    [(Type) "Type"]
    [(Nat) "Nat"]
    [(Nat-ind e1 e2 e3 e4)
       (format "(nat-ind ~a ~a ~a ~a)"
               (pretty-print-expr e1) (pretty-print-expr e2) (pretty-print-expr e3) (pretty-print-expr e4))]
    [(Z) "Z"]
    [(S x) (format "(S ~a)" (pretty-print-expr x))]
    [(Eq-refl a) (format "(eq-refl ~a)" (pretty-print-expr a))]
    [(Eq-elim e1 e2 e3 e4 e5)
      (format "(eq-elim ~a ~a ~a ~a ~a)" (pretty-print-expr e1) (pretty-print-expr e2)
              (pretty-print-expr e3) (pretty-print-expr e4) (pretty-print-expr e5))]
    [(Teq e1 e2) (format "(~a = ~a)" (pretty-print-expr e1) (pretty-print-expr e2))]))

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
    [(Type) (Type)]
    [(Nat) (Nat)]
    [(Nat-ind e1 e2 e3 e4)
       (Nat-ind (subst oldx newx e1) (subst oldx newx e2) (subst oldx newx e3) (subst oldx newx e4))]
    [(Z) (Z)]
    [(S x) (S (subst oldx newx x))]
    [(Eq-refl a) (Eq-refl (subst oldx newx a))]
    [(Eq-elim e1 e2 e3 e4 e5)
      (Eq-elim (subst oldx newx e1) (subst oldx newx e2) (subst oldx newx e3)
               (subst oldx newx e4) (subst oldx newx e5))]
    [(Teq a b) (Teq (subst oldx newx a) (subst oldx newx b))]))

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
    [(Type) false]
    [(Nat) false]
    [(Nat-ind e1 e2 e3 e4)
       (side-cond? x (list e1 e2 e3 e4) check-binders?)]
    [(Z) false]
    [(S a) (sc-helper x a check-binders?)]
    [(Eq-refl a) (sc-helper x a check-binders?)]
    [(Eq-elim e1 e2 e3 e4 e5)
      (side-cond? x (list e1 e2 e3 e4 e5) check-binders?)]
    [(Teq a b) (side-cond? x (list a b) check-binders?)]))

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
    [(list (Nat) (Nat)) true]
    [(list (Z) (Z)) true]
    [(list (S x1) (S x2)) (ae-helper x1 x2 vmap)]
    [(list (Eq-refl x1) (Eq-refl x2)) (ae-helper x1 x2 vmap)]
    [(list (Eq-elim x1 P1 px1 y1 peq1) (Eq-elim x2 P2 px2 y2 peq2))
       (and (ae-helper x1 x2 vmap) (ae-helper P1 P2 vmap) (ae-helper px1 px2 vmap)
            (ae-helper y1 y2 vmap) (ae-helper peq1 peq2 vmap))]
    [(list (Teq a1 b1) (Teq a2 b2))
       (and (ae-helper a1 a2 vmap) (ae-helper b1 b2 vmap))]
    [(list (Nat-ind e11 e12 e13 e14) (Nat-ind e21 e22 e23 e24))
       (and (ae-helper e11 e21 vmap) (ae-helper e12 e22 vmap) (ae-helper e13 e23 vmap) (ae-helper e14 e24 vmap))]
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
    [(Type) (Type)]
    [(Nat) (Nat)]
    [(Nat-ind P zc sc n)
       (define nr (reduce ctx n))
       (match nr
         [(Z) (reduce ctx zc)]
         ; ================================================================
         ; ================================================================
         [(S m) (reduce ctx (App (App sc m) (Nat-ind P zc sc m)))]
         [else (Nat-ind (reduce ctx P) (reduce ctx zc) (reduce ctx sc) nr)])]
    [(Z) (Z)]
    [(S x) (S (reduce ctx x))]
    [(Eq-refl x) (Eq-refl (reduce ctx x))]
    [(Eq-elim x P px y peq)
       (define peqr (reduce ctx peq))
       (match peqr
         [(Eq-refl _) (reduce ctx px)]
         [else (Eq-elim (reduce ctx x) (reduce ctx P) (reduce ctx px) (reduce ctx y) peqr)])]
    [(Teq a b) (Teq (reduce ctx a) (reduce ctx b))]))

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
    [(Type) (Type)]
    [(Nat) (Type)]
    [(Nat-ind P zc sc n)
       (type-check ctx P (Arrow '_ (Nat) (Type)))
       (define Pann (Ann P (Arrow '_ (Nat) (Type))))
       (type-check ctx zc (App Pann (Z)))

       ; ================================================================================
       ; ================================================================================
       
       ;  Notice in particular that if you are constructing the expected type of sc,
       ; the variable of quantification k should be fresh with respect to P.
       (define k (refresh 'k (list P)))

       ;(type-check ctx fp (App Pann (False)))
       (type-check ctx sc (Arrow k (Nat) (Arrow '_ (App Pann k) (App Pann (S k)))))
      
       (type-check ctx n (Nat))
       (App Pann n)]
    [(Z) (Nat)]
    [(S x) (type-synth ctx x)]
    [(Teq e1 e2)
       (define t1 (type-synth ctx e1))
       (type-check ctx e2 t1)
       (Type)]
    [(Eq-refl x) (type-synth ctx x) (Teq x x)]
    [(Eq-elim x P px y peq)
       (define A (type-synth ctx x))
       (type-check ctx P (Arrow '_ A (Type)))
       (define Pann (Ann P (Arrow '_ A (Type))))
       (type-check ctx px (App Pann x))
       (type-check ctx y A)
       (type-check ctx peq (Teq x y))
       (App Pann y)]))

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


; ================================================================================
; ================================================================================


(def 'zero '(z : Nat))
(def 'one '((s z): Nat))
(def 'two '((s (s z)): Nat))
(def 'succ '((λ n => (s n)) : (Nat -> Nat)))


(def 'nat-iter '((λ C => (λ zc => (λ sc => (λ n => (nat-ind (λ x => C) zc (λ x => sc) n)))))
                 :(∀ (C : Type) -> (C -> ((C -> C) -> (Nat -> C))))))

(def 'plus '((λ n => (λ m => (nat-iter Nat m succ n)))
             :(Nat -> (Nat -> Nat))))

(def 'P-plus-zero-left '((λ x => ((plus z x) = x))
                         :(Nat -> Type)))
(def 'plus-zero-left '((λ n => (nat-ind P-plus-zero-left (eq-refl z) (λ x => (λ sc => (eq-refl (s x)))) n))
                       : (∀ (n : Nat) -> (P-plus-zero-left n))))

; eq-sym
(def 'eq-sym '((λ x => (λ y => (λ xy => (eq-elim x (λ b => (b = x)) (eq-refl x) y xy))))
               :(∀ (x : Type) -> (∀ (y : Type) -> ((x = y) -> (y = x))))))

; eq-tran
(def 'eq-tran '((λ a => (λ b => (λ c => (λ ab => (λ bc => (eq-elim b (λ x => (x = c)) bc a (((eq-sym a) b) ab)))))))
                :(∀ (a : Type) -> (∀ (b : Type) -> (∀ (c : Type) -> ((a = b) -> ((b = c) -> (a = c))))))))

; (x = y) -> (sx = sy)
(def 'lemma5 '((λ n => (λ m => (λ nm => (eq-elim n (λ x => ((succ n) = (succ x))) (eq-refl (succ n)) m nm))))
                : (∀ (n : Nat) -> (∀ (m : Nat) -> ((n = m) -> ((succ n) = (succ m)))))))

; ex 11
(def 'P-plus-zero-right '((λ n => ((plus n z) = n))
                          :(Nat -> Type)))
(def 'plus-zero-right '((λ n => (nat-ind P-plus-zero-right (eq-refl z) (λ x => (λ sc => (((lemma5 ((plus x) z)) x) sc))) n))
                        : (∀ (n : Nat) -> (P-plus-zero-right n))))

; add logical connectives
; peano axioms from stewart. show that they can all be proved in proust-pred-nat
































