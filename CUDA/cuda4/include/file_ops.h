#ifndef FILE_OPS_H
#define FILE_OPS_H


// concept: the init file contains a number N, which says there 
// are N photos & params existing. this allows us to then load 
// any of the N params. and when we create a new photo/param, 
// we know to call it N+1, and update the init file to say N+1.


class file_manager
{
public:
    // the init file name
    const char* init_file_name = "fractal_init";

    // check if file exists.
    int file_exists(const char *filename);

    // write an integer to the init file
    void write_int_to_init_file(int x);

    // if init file exists, read it, otherwise return 0
    int read_int_from_init_file();
};


#endif