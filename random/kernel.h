#include <stdio.h>
#define UNIX YES
/* scalar constants: */
#define HASHTABSIZE 256 /* size of hash table */
#define CELLTABSIZE 1024 /* size of cell table */
#define BLOCKSIZE 512 /* block size for cell allocation */
#define EVALSTKSIZE 1024 /* size of evaluation stack */
#define VARSTKSIZE 1024 /* size of variable stack */
#define ARGSTKSIZE 1024 /* size of argument stack */
#define CATSTKSIZE 256 /* size of catch stack */
#define CHANBUFSIZE 126 /* size of channel buffer */
#define STRBUFSIZE 126 /* size of string buffer */
#define MAXCOLS 80 /* max no. of columns on the screen */
#define SMALLINTLOW -128 /* least small integer */
#define SMALLINTHIGH 127 /* greatest small integer */

/* values for flag in symbol structure: */
#define UNBOUND 0 /* unbound symbol */
#define CONSTANT 1 /* constant - cannot be changed */
#define VARIABLE 2 /* bound variable */
#define FUNCTION 3 /* non-binary function */
#define LBINARY 4 /* binary lam */
#define VBINARY 5 /* binary vlam */
#define UBINARY 6 /* binary ulam */
#define MBINARY 7 /* binary mlam */
#define INTERNAL 8 /* internal object - not in sy.nbol table */

/* values for flag in cell structure: */
#define VOID 10 /* void object */
#define INTOBJ 11 /* integer number */
#define REALOBJ 12 /* real number */
#define STROBJ 13 /* string */
#define CHANOBJ 14 /* channel for I/O */
#define VECTOROBJ 15 /* vector */
#define LISTOBJ 16 /* list */
#define SETOBJ 17 /* set */
#define MARK 128 /* mark bit - for garbage collection */
#define MASK7 127 /* for masking bit 7 in flag */

/* channel kinds: */
#define INCHAN 0 /* input channel flag */
#define OUTCHAN 1 /* output channel flag */
#define INOUTCHAN 2 /* input-output channel flag */

/* values for flag in printaux and bufprint: */
#define PRINT 0 /* flag = PRINT => print */
#define PRINC 1. /* flag = PRINC => princ */
#define LENGTH 2 /* flag = LENGTH => prlen */
#define STRIP 3 /* |symbol| => symbol */

typedef unsigned char byte; /* the basic byte unit */
typedef union {int i,*j;} word; /* the basic word unit */
typedef float real; /* real type - can be changed to double */

struct symbol { /* symbol structure */
    byte flag; /* symbol type, always < VOID */
    struct cell *bind; /* symbol binding */
    struct cell *prop; /* symbol property list */
    char *name; /* symbol name */
    struct symbol *link; /* link to next symbol */
}

struct cell { /* cons-cell structure */
    byte flag; /* cell type, always >= VOID */
    union {
        int inum; /* integer number */
        real rnum; /* veal number */
        char *str; /* string */
        struct channel *chan; /* channel */
        struct { /* for list/set construction */
            struct cell *car; /* car pointer */
            struct cell *cdr; /* cdr pointer */
        } pair;
        struct { /* for vector construction */
            struct cell *dim; /* vector dimension */
            struct cell **vec; /* vector block */
        } vect;
    } part;
}

struct channel { /* I/O channel structure */
    char ch; /* current character */
    unsigned short int tok; /* current token */
    unsigned short int pos; /* current position in buf */
    unsigned short int len; /* no. of chars in buf */
    char *buf; /* channel buffer */
    byte mode; /* one of INCHAN,OUTCHAN, INOUTCHAN */
    FILE *file; /* the file associated with channel */
}

struct variable { /* variable structure for variable stack */
    struct symbol *sym; /* variable symbol */
    byte flag; /* its flag */
    struct cell *bind; /* its binding */
}













typedef struct symbol *kernsym; /* symbol pointer */
typedef struct cell *kerncell; /* cell pointer */
typedef struct channel *iochan; /* I/O channel */

/* macros: */
#define ISnotbinary(p)  ((p)->flag <  LBINARY)
#define ISunbound(p)    ((p)->flag == UNBOUND)
#define ISconst(p)      ((p)->flag == CONSTANT)
#define ISvar(p)        ((p)->flag == VARIABLE)
#define ISfun(p)        ((p)->flag == FUNCTION)
#define ISlbin(p)       ((p)->flag == LBINARY)
#define ISvbin(p)       ((p)->flag == VBINARY)
#define ISubin(p)       ((p)->flag == UBINARY)
#define ISmbin(p)       ((p)->flag == MBINARY)
#define ISinternal(p)   ((p)->flag == INTERNAL)

#define ISsym(p)        ((p)->flag <  VOID)
#define IScel1(p)       ((p)->flag >= VOID)
#define ISvoid(p)       ((p)->flag == VOID)
#define ISint(p)        ((p)->flag == INTOBJ)
#define ISreal(p)       ((p)->flag == REALOBJ)
#define ISstr(p)        ((p)->flag == STROBJ)
#define ISchan(p)       ((p)->flag == CHANOBJ)
#define ISvector(p)     ((p)->flag == VECTOROBJ)
#define ISlist(p)       ((p)->flag >= LISTOBJ)
#define ISset(p)        ((p)->flag == SETOBJ)
#define ISmarked(p)     (((p)->flag & MARK) == MARK)

#define CELLinum   part.inum
#define CELLrnum   part.rnum
#define CELLstr    part.str
#define CELLchan   part.chan
#define CELLcar    part.pair.car
#define CELLcdr    part.pair.cdr
#define CELLdim    part.vect.dim
#define CELLvec    part.vect.vec

#define CONVbyte(p)   ((byte) (p))
#define CONVint(p)    ((int) (p))
#define CONVintp(p)   ((int *) (p))
#define CONVreal(p)   ((real) (p))
#define CONVstr(p)    ((char *) (p))
#define CONVchan(p)   ((iochan) (p) )
#define CONVsym(p)    ((kernsym) (p))
#define CONVcelL(p)   ((kerncell) (p))
#define CONVvector(p) ((kerncell *) (p))
#define NIL           ((kerncell) nil)
#define TtT           ((kerncell) ttt)

#define READin() readaux(_inchan, 0)
#define READchan(chan) readaux((chan)->CELLchan,0)
#define PRINTout(p) printaux(PRINT, (p),_outchan)
#define PRINTchan(p,chan) printaux(PRINT, (p), (chan)->CELLchan)
#define TERPRIout() bufprint(PRINT, _outchan,"\n")
#define TERPRIchan(chan) bufprint(PRINT, (chan)->CELLchan, "\n")
#define INTERNALsym(isym) \
    (isym = CONVsym(new(sizeof(struct symbol))))—>flag = INTERNAL










// PAUSED HERE














#define CHECKlargs (fun,n) \
    if (argtop - CONVint (argstk[argtop]) != (n)) \
        error (fun,err_args,0)
#define CHECKvargs (fun,m,n) x
    if (argtop - CONVint (argstk[argtop]) < (m) || \
            argtop - CONVint (argstk[argtop]) > (n)) \
        error (fun,err_args,0)
#define CHECKvargs1 (fun,n) *
    if (argtop — CONVint (argstk[argtop]) < (n)) \
error (fun,err_args,0)
#define CHECKvargs2 (fun,n) \
if (argtop - CONVint (argstk[argtop]) > (n)) \
error (fun,err_args,0)
#define EVALpush (ob}) \
(+Hevaltop < celltop ? evalstk[evaltop] = (obj) nN
: CONVcell (faterr (err_evalstk) ))
#define EVALpop () --evaltop
#define CELLpush (ob}) x
(--celltop > evaltop ? evalstk[celltop] = (obj) x
: CONVcell (faterr (err_evalstk) ) )
#define CELLpop() +t+celltop
#define VARpush(s, f,b) \
if (++vartop < VARSTKSIZE) { i‘
varstk[vartop].sym = (s); \
varstk[vartop].flag = (f); \
varstk[vartop] .bind = (b); \
} else faterr(err_varstk)
#define VARpop () \
{ varstk[vartop].sym->flag = varstk[vartop] .flag; *
varstk[vartop] .sym->bind = varstk[vartop--] .bind; }
#define ARGpush (obj) \
if (Hargtop < EVALSTKSIZE) argstk[argtop] = (ob}); \
else faterr (err_argstk)
#define ARGpop () —~argtop
#define ARGSpop () argtop = CONVint (argstk[argtop]) - 1
#define ARGidxl CONVint (argstk [argtop] )
#define ARGnum1l argstk [ARGidx1]
