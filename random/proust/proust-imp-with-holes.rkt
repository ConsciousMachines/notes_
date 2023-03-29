#lang racket

(struct Lam (var body) #:transparent)
(struct App (rator rand) #:transparent)
(struct Ann (expr type) #:transparent)
(struct Hole (num) #:transparent) ; arbitrary numbering
(struct ∧-intro (e1 e2) #:transparent)
(struct ∧-elim0 (e) #:transparent)
(struct ∧-elim1 (e) #:transparent)
(struct ∨-intro0 (e) #:transparent)
(struct ∨-intro1 (e) #:transparent)
(struct ∨-elim (e1 e2 e3) #:transparent)

(struct Arrow (domain range) #:transparent)
(struct And (t1 t2) #:transparent)
(struct Or (t1 t2) #:transparent)

(define (parse-expr s)
  (match s
    [`(∨-elim ,e1, e2 ,e3) (∨-elim (parse-expr e1) (parse-expr e2) (parse-expr e3))]
    [`(∨-intro1 ,e) (∨-intro1 (parse-expr e))]
    [`(∨-intro0 ,e) (∨-intro0 (parse-expr e))]
    [`(∧-elim1 ,e) (∧-elim1 (parse-expr e))]
    [`(∧-elim0 ,e) (∧-elim0 (parse-expr e))]
    [`(∧-intro ,e1 ,e2) (∧-intro (parse-expr e1) (parse-expr e2))]
    [`(λ ,(? symbol? x) => ,e) (Lam x (parse-expr e))]
    [`(,e1 ,e2) (App (parse-expr e1) (parse-expr e2))]
    ['? (Hole #f)] ; not numbered yet
    [(? symbol? x) x]
    [`(,e : ,t) (Ann (parse-expr e) (parse-type t))]
    [`(,e1 ,e2 ,e3 ,r ...) (parse-expr `((,e1 ,e2) ,e3 ,@r))]
    [else (error 'parse "bad syntax: ~a" s)]))

(define (parse-type t)
  (match t
    [`(,t1 ∨ ,t2) (Or (parse-type t1) (parse-type t2))]
    [`(,t1 ∧ ,t2) (And (parse-type t1) (parse-type t2))]
    [`(,t1 -> ,t2) (Arrow (parse-type t1) (parse-type t2))]
    [`(,t1 -> ,t2 -> ,r ...) (Arrow (parse-type t1) (parse-type `(,t2 -> ,@r)))]
    [(? symbol? X) X] 
    [else (error 'parse "bad syntax: ~a\n" t)]))

(define (pretty-print-expr e)
  (match e
    [(∨-elim e1 e2 e3) (format "(∨-elim ~a ~a ~a)" (pretty-print-expr e1) (pretty-print-expr e2) (pretty-print-expr e3))]
    [(∨-intro1 e) (format "(∨-intro1 ~a)" (pretty-print-expr e))]
    [(∨-intro0 e) (format "(∨-intro0 ~a)" (pretty-print-expr e))]
    [(∧-elim1 e) (format "(∧-elim1 ~a)" (pretty-print-expr e))]
    [(∧-elim0 e) (format "(∧-elim0 ~a)" (pretty-print-expr e))]
    [(∧-intro e1 e2) (format "(∧-intro ~a ~a)" (pretty-print-expr e1) (pretty-print-expr e2))]
    [(Lam x b) (format "(λ ~a => ~a)" x (pretty-print-expr b))]
    [(App e1 e2) (format "(~a ~a)" (pretty-print-expr e1) (pretty-print-expr e2))]
    [(? symbol? x) (format "~a" x)]
    [(Ann e t) (format "(~a : ~a)" (pretty-print-expr e) (pretty-print-type t))]
    [(Hole n) (format "?~a" n)]))

(define (pretty-print-type t)
  (match t
    [(Or t1 t2) (format "(~a ∨ ~a)" (pretty-print-type t1) (pretty-print-type t2))]
    [(And t1 t2) (format "(~a ∧ ~a)" (pretty-print-type t1) (pretty-print-type t2))]
    [(Arrow t1 t2) (format "(~a -> ~a)" (pretty-print-type t1) (pretty-print-type t2))]
    [else t]))

(define (pretty-print-context ctx)
  (cond
    [(empty? ctx) ""]
    [else (string-append (format "\n~a : ~a" (first (first ctx)) (pretty-print-type (second (first ctx))))
                         (pretty-print-context (rest ctx)))]))

(define (type-check ctx expr type)
  ;(display (pretty-print-type type))
  (match expr
    [(∨-elim e1 e2 e3)
       (define t0 (type-synth ctx e1))
       (match t0
         [(Or t1 t2) (and (type-check ctx e2 (Arrow t1 type)) (type-check ctx e3 (Arrow t2 type)))]
         [else (error "expected OR type in or-elim")])]
    [(∨-intro1 e)
       (match type
         [(Or t1 t2) (type-check ctx e t2)]
         [else (error "type of or-intro1 is not OR")])]
    [(∨-intro0 e)
       (match type
         [(Or t1 t2) (type-check ctx e t1)]
         [else (error "type of or-intro0 is not OR")])]
    [(∧-intro e1 e2)
       (match type
         [(And t1 t2) (and (type-check ctx e1 t1) (type-check ctx e2 t2))]
         [else (error "type of AND expression is not an AND type")])]
    [(Lam x t) 
       (match type
         [(Arrow tt tw) (type-check (cons `(,x ,tt) ctx) t tw)]
         [else (error "bruh")])]
    [(Hole n) (when refining (hash-set! goal-table n (list type ctx)))
              true]
    [else (if (equal? (type-synth ctx expr) type) true (error "bruh"))]))

(define (type-synth ctx expr)
  (match expr
    [(∧-elim1 e)
       (define t0 (type-synth ctx e))
       (match t0
         [(And t1 t2) t2]
         [else (error "expected AND type")])]
    [(∧-elim0 e)
       (define t0 (type-synth ctx e))
       (match t0
         [(And t1 t2) t1]
         [else (error "expected AND type")])]
    [(Lam _ _) (error "bruh")]
    [(Hole _) (error "bruh")]
    [(Ann e t) (type-check ctx e t) t]
    [(App f a)
       (define tf (type-synth ctx f))
        (match tf
         [(Arrow tt tw) #:when (type-check ctx a tt) tw]
         [else (error "bruh")])]
    [(? symbol? x)
       (cond
         [(assoc x ctx) => second]
         [else (error "didnt find type in the context:" x)])]))












(define current-expr #f)
(define goal-table (make-hash))
(define hole-ctr 0)
(define (use-hole-ctr) (begin0 hole-ctr (set! hole-ctr (add1 hole-ctr))))
(define refining #f)


(define (set-task! s)
  (define t (parse-type s))
  (set! goal-table (make-hash))
  (set! hole-ctr 1)
  (set! current-expr (Ann (Hole 0) t))
  (hash-set! goal-table 0 (list t empty))
  (printf "\nTask is now\n")
  (print-task))

(define (print-task) (printf "~a\n" (pretty-print-expr current-expr)))

(define print-goal
  (case-lambda
    [() (hash-for-each goal-table (lambda (n g) (print-goal n)))]
    [(n) (match-define (list t ctx)
           (hash-ref goal-table n (lambda () (error 'refine "no goal of that number"))))
         (printf "Goal ~a has type ~a\n" n (pretty-print-type t))
         (unless (empty? ctx)
           (printf "in context~a\n" (pretty-print-context ctx)))]))

(define (refine n s)
  (match-define (list t ctx)
    (hash-ref goal-table n (lambda () (error 'refine "no goal numbered ~a" n))))
  (define e (parse-expr s))
  (type-check ctx e t) ; first time, just check
  (define en (number-new-holes e))
  (set! refining #t)
  (type-check ctx en t) ; second time, add new goals to goal table
  (set! refining #f)
  (hash-remove! goal-table n)
  (set! current-expr (replace-goal-with n en current-expr))
  (define ngoals (hash-count goal-table))
  (printf "\nTask with ~a is now\n" (format "~a goal~a" ngoals (if (= ngoals 1) "" "s")))
  (print-task)
  (print-goal))

(define (replace-goal-with n repl e)
  (match e
    [(Lam x b) (Lam x (replace-goal-with n repl b))]
    [(App e1 e2) (App (replace-goal-with n repl e1) (replace-goal-with n repl e2))]
    [(? symbol? x) x]
    [(Ann e t) (Ann (replace-goal-with n repl e) t)]
    [(Hole m) (if (= m n) repl (Hole m))]))

(define (number-new-holes e)
  (match e
    [(∨-elim e1 e2 e3) (∨-elim (number-new-holes e1) (number-new-holes e2) (number-new-holes e3))]
    [(∨-intro0 e) (∨-intro0 (number-new-holes e))]
    [(∨-intro1 e) (∨-intro1 (number-new-holes e))]
    [(∧-elim1 e) (∧-elim1 (number-new-holes e))]
    [(∧-elim0 e) (∧-elim0 (number-new-holes e))]
    [(∧-intro e1 e2) (∧-intro (number-new-holes e1) (number-new-holes e2))]
    [(Lam x b) (Lam x (number-new-holes b))]
    [(App e1 e2) (App (number-new-holes e1) (number-new-holes e2))]
    [(? symbol? x) x]
    [(Ann e t) (Ann (number-new-holes e) t)]
    [(Hole m) (if m (Hole m) (Hole (use-hole-ctr)))]))

(define (curr-sexp) (with-input-from-string (pretty-print-expr current-expr) read))

(define example1
 (begin
  (set-task! '(A -> (A -> B) -> B))
  (refine 0 '(λ x => ?))
  (refine 1 '(λ y => ?))
  (refine 2 '(y x))
  (curr-sexp)))

(define example2
 (begin
  (set-task! '(A -> B -> A))
  (refine 0 '(λ x => ?))
  (refine 1 '(λ y => ?))
  (refine 2 'x)
  (curr-sexp)))

(display "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

(set-task! '((A -> B -> C) -> (A -> B) -> (A -> C)))
(refine 0 '(λ a => ?))
(refine 1 '(λ b => ?))
(refine 2 '(λ c => ?))
(refine 3 '((a c) (b c)))

(set-task! '(((A -> B) -> (A -> C)) -> (A -> B -> C)))
(refine 0 '(λ a => ?))
(refine 1 '(λ b => ?))
(refine 2 '(λ c => ?))
(refine 3 '((a (λ b => c)) b))

(set-task! '((B -> C) -> (A -> B) -> (A -> C)))
(refine 0 '(λ a => ?))
(refine 1 '(λ b => ?))
(refine 2 '(λ c => ?))
(refine 3 '(a (b c)))

(set-task! '(A -> (B -> (A ∧ B))))
(refine 0 '(λ a => ?))
(refine 1 '(λ b => ?))
(refine 2 '(∧-intro a b))

(set-task! '(((A ∧ B) -> C) -> (A -> (B -> C))))
(refine 0 '(λ a => ?))
(refine 1 '(λ b => ?))
(refine 2 '(λ c => ?))
(refine 3 '(a (∧-intro b c)))

(set-task! '((A ∧ B) -> A))
(refine 0 '(λ a => ?))
(refine 1 '(∧-elim0 a))

(set-task! '((A ∧ B) -> (B ∧ A)))
(refine 0 '(λ a => ?))
(refine 1 '(∧-intro (∧-elim1 a) (∧-elim0 a)))

(set-task! '((A -> (B -> C)) -> ((A ∧ B) -> C)))
(refine 0 '(λ a => ?))
(refine 1 '(λ b => ?))
(refine 2 '((a (∧-elim0 b)) (∧-elim1 b)))

(set-task! '((A -> B) -> ((A ∧ C) -> (B ∧ C))))
(refine 0 '(λ a => ?))
(refine 1 '(λ b => ?))
(refine 2 '(∧-intro (a (∧-elim0 b)) (∧-elim1 b)))

(set-task! '(((A -> B) ∧ (C -> D)) -> ((A ∧ C) -> (B ∧ D))))
(refine 0 '(λ a => ?))
(refine 1 '(λ b => ?))
(refine 2 '(∧-intro ((∧-elim0 a) (∧-elim0 b)) ((∧-elim1 a) (∧-elim1 b))))

(set-task! '(A -> (A ∨ B)))
(refine 0 '(λ a => ?))
(refine 1 '(∨-intro0 a))

(set-task! '((A ∨ A) -> A))
(refine 0 '(λ d => ?))
(refine 1 '(∨-elim d (λ a => a) (λ a => a)))

;(define e13 '((A ∨ B) -> (B ∨ A)))
;(set-task! e13)
;(refine 0 '(λ a => ?))
