typedef struct __labl
{
    struct __labl* next;
    struct __labl* prev;
    char name[ MAX_CHAR ];
    int addr;
} labl;


typedef struct __smnt
{
    int labl_n;
    char labl_s[ MAX_LABL ][ MAX_CHAR ];
    int mnem_n;
    char mnem_s[ MAX_MNEM ][ MAX_CHAR ];
    int oper_n;
    char oper_s[ MAX_OPER ][ MAX_CHAR ];
} smnt;


typedef struct __inst
{
    char mnem_s[ MAX_CHAR ];
    int code;
    int oper_n;
    int oper_t[ MAX_OPER ];
} inst;

inst tab_inst[] = {
    {"add",     CODE_ADD,     3, {OPER_RD, OPER_RS,OPER_RT       }},   
    {"sub",     CODE_SUB,     3, {OPER_RD, OPER_RS,OPER_RT       }},  
    {"mul",     CODE_MUL,     3, {OPER_RD, OPER_RS,OPER_RT       }},   
    {"mult",    CODE_MULT,    2, {         OPER_RS,OPER_RT       }},   
    {"and",     CODE_AND,     3, {OPER_RD, OPER_RS,OPER_RT       }},       
    {"or",      CODE_OR,      3, {OPER_RD, OPER_RS,OPER_RT       }},      
    {"xor",     CODE_XOR,     3, {OPER_RD, OPER_RS,OPER_RT       }},  
    {"nor",     CODE_NOR,     3, {OPER_RD, OPER_RS,OPER_RT       }}, 
    {"sllv",    CODE_SLLV,    3, {OPER_RD, OPER_RS,OPER_RT       }}, 
    {"srlv",    CODE_SRLV,    3, {OPER_RD, OPER_RS,OPER_RT       }}, 
    {"addi",    CODE_ADDI,    3, {OPER_RT, OPER_RS,OPER_IMM16    }},
                                                       
    {"mfhi",    CODE_MFHI,    1, {OPER_RD                        }},       
    {"mflo",    CODE_MFLO,    1, {OPER_RD                        }},      
    {"mfc0",    CODE_MFC0,    2, {OPER_RD, OPER_RT               }},     
    {"mthi",    CODE_MTHI,    1, {OPER_RS                        }},     
    {"mtlo",    CODE_MTLO,    1, {OPER_RS                        }},     
    {"mtc0",    CODE_MTC0,    2, {OPER_RD, OPER_RT               }},     
    {"lw",      CODE_LW,      3, {OPER_RT, OPER_IMM16, OPER_RS   }},  
    {"sw",      CODE_SW,      3, {OPER_RT, OPER_IMM16, OPER_RS   }},
                                            
    {"beq",     CODE_BEQ,     3, {OPER_RS, OPER_RT, OPER_OFF16_WA}},
    {"bne",     CODE_BNE,     3, {OPER_RS, OPER_RT, OPER_OFF16_WA}}, 
    {"blez",    CODE_BLEZ,    2, {OPER_RS,          OPER_OFF16_WA}},      
    {"bgtz",    CODE_BGTZ,    2, {OPER_RS,          OPER_OFF16_WA}},        
    {"bltz",    CODE_BLTZ,    2, {OPER_RS,          OPER_OFF16_WA}},        
    {"bgez",    CODE_BGEZ,    2, {OPER_RS,          OPER_OFF16_WA}},         
    {"b",       CODE_B,       1, {                  OPER_OFF26_WA}},          
    {"j",       CODE_J,       1, {                  OPER_IMM26_WA}},    
    {"jr",      CODE_JR,      1, {OPER_RS                        }},             
    {"jal",     CODE_JAL,     1, {                  OPER_IMM26_WA}},     
                                                   
    {"nop",     CODE_NOP,     0, {                               }},
    {"syscall", CODE_SYSCALL, 0, {                               }},
    {"break",   CODE_BREAK,   0, {                               }}
};


void parse( FILE* f, void ( *process )( smnt* s ) )
{
    char l[ MAX_CHAR ];
    smnt s;

    while( fgets( l, MAX_CHAR, f ) != NULL )
    {
        char* p;
        char* q;


        for( int i = strlen( l ) - 1; i >= 0; i-- )
        {
            if( l[ i ] == ’\n’ || l[ i ] == ’#’ )
                l[ i ] = ’\0’;
            if( l[ i ] == ’\t’ || l[ i ] == ’,’ || l[ i ] == ’(’ || l[ i ] == ’)’ )
                l[ i ] = ’ ’;
        }


        s.labl_n = 0;
        s.mnem_n = 0;
        s.oper_n = 0;


    for( q = l; ( ( p = strtok( q, " " ) ) != NULL ); q = NULL )
    {
        if( s.mnem_n == 0 )
        {
            if( p[ strlen( p ) - 1 ] == ’:’ )
                strcpy( s.labl_s[ s.labl_n++ ], p );
            else
                strcpy( s.mnem_s[ s.mnem_n++ ], p );
        }
        else
        {
            strcpy( s.oper_s[ s.oper_n++ ], p );
        }
    }

    process( &s );
}







void pass_one_directive( smnt* s )
{
    if( !strcmp( s->mnem_s[ 0 ], ".word" ) )
    {
        for( int i = 0; i < s->oper_n; i++ )
        {
            int n = atoi( s->oper_s[ i ] );
            addr += 4;
        }
    }
}



void pass_one_instruction( smnt* s )
{
    inst* t = get_inst( s->mnem_s[ 0 ] );
    if ( t == NULL )
        error( "unknown instruction" );
    else if( t->oper_n != s->oper_n )
        error( "wrong operand count" );
    else
    {
        for( int i = 0; i < t->oper_n; i++ )
        {
            switch( t->oper_t[ i ] )
            {
            case OPER_RD:
            case OPER_RT:
            case OPER_RS:
                if( !is_reg( s->oper_s[ i ] ) )
                    error( "wanted register operand" );
                break;
            case OPER_IMM16 :
            case OPER_OFF16 :
            case OPER_IMM26 :
            case OPER_OFF26 :
                if( !is_num( s->oper_s[ i ] ) )
                    error( "wanted numeric operand" );
                break;
            }
        }
        addr += 4;
    }
}



void pass_one_statement( smnt* s )
{
    if( s->labl_n != 0 )
    {
        if( get_labl( s->labl_s[ 0 ] ) != NULL )
            error( "redefined label" );
        else
            add_labl( s->labl_s[ 0 ], addr );
    }
    if( s->mnem_n != 0 )
    {
        if( is_directive( s->mnem_s[ 0 ] ) )
            pass_one_directive( s );
        else
            pass_one_instruction( s );
    }
}

















void pass_two_directive( smnt* s )
{
    if( !strcmp( s->mnem_s[ 0 ], ".word" ) )
    {
        for( int i = 0; i < s->oper_n; i++ )
        {
            int n = atoi( s->oper_s[ i ] );
            output( addr, n );
            addr += 4;
        }
    }
}


void pass_two_instruction( smnt* s )
{
    inst* t = get_inst( s->mnem_s[ 0 ] );
    uint32 enc = t->code;
    for( int i = 0; i < t->oper_n; i++ )
    {
        switch( t->oper_t[ i ] )
        {
        case OPER_RD:
            enc |= ( ( dec_reg( s->oper_s[ i ] )) & 0x0000003F ) << 11;
            break;
        case OPER_RT:
            enc |= ( ( dec_reg( s->oper_s[ i ] )) & 0x0000003F ) << 16;
            break;
        case OPER_RS:
            enc |= ( ( dec_reg( s->oper_s[ i ] )) & 0x0000003F ) << 21;
            break;
        case OPER_IMM16:
            enc |= ( ( dec_num( s->oper_s[ i ] )) & 0x0000FFFF );
            break;
        case OPER_OFF16:
            enc |= ( ( dec_num( s->oper_s[ i ] ) - addr - 4 ) & 0x0000FFFF );
            break;
        case OPER_IMM26:
            enc |= ( ( dec_num( s->oper_s[ i ] )) & 0x03FFFFFF );
            break;
        case OPER_OFF26:
            enc |= ( ( dec_num( s->oper_s[ i ] ) - addr - 4 ) & 0x03FFFFFF );
            break;
        case OPER_IMM16_WA :
            enc |= ( ( dec_num( s->oper_s[ i ] )) & 0x0000FFFF ) >> 2;
            break;
        case OPER_OFF16_WA :
            enc |= ( ( dec_num( s->oper_s[ i ] ) - addr - 4 ) & 0x0000FFFF ) >> 2;
            break;
        case OPER_IMM26_WA :
            enc |= ( ( dec_num( s->oper_s[ i ] )) & 0x03FFFFFF ) >> 2;
            break;
        case OPER_OFF26_WA :
            enc |= ( ( dec_num( s->oper_s[ i ] ) - addr - 4 ) & 0x03FFFFFF ) >> 2;
            break;
        }
    }
    output( addr, enc );
    addr += 4;
}


void pass_two_statement( smnt* s )
{
    if( s->mnem_n != 0 )
    {
        if( is_directive( s->mnem_s[ 0 ] ) )
            pass_two_directive( s );
        else
            pass_two_instruction( s );
    }
}