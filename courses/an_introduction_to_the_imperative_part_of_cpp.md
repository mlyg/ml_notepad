# An Introduction to the Imperative Part of C++

## Lecture 1
1. **Declarative programming**: what to do, not how to do it - higher level of abstraction
2. **Imperative programming**: how to do it, not what to do - lower level of abstraction
3. **Functional programming**: subset of declarative programming, that uses subroutines that return a value, and cause minimal side effects
4. **Procedural programming**: subset of imperative programming, that uses subroutines that do not return values, causing side effects
5. C++ was developed by **Bjarne Stroustrup** in the early 1980's
6. A **compiler** converts the source code into machine instructions
7. A **linking program** links the compiled program components with each other and with a selection of routines from existing libraries of computer code

## Lecture 2
1. **Words in a programme belong to three types:**
* **Reserved words**: These are words such as if, int and else, which have a predefined meaning that cannot be changed
* **Library identifiers**: These words are supplied default meanings by the programming environment such as cin, cout and sqrt
* **Programmer-supplied identifiers**: variable names created by the programmer
2. C++ requires that all variables used in a program be given a data type
3. **Prefixes to control base of number:**
* "0": **octal**
* "0x": **hexadecimal**
* "0b": **binary**
4. The data type "char" is a subset of the data type "int"
5. Constants of type "int" may also be declared with an **enumeration** statement
6. The operator "&&" has a higher precedence than the operator "||"

## Lecture 3
1. A function has to be declared in a function declaration at the top of the program
2. **Function declarations** specify which type the function returns
3. While **value parameters** are safe, **reference parameters** are specified with "&" postfix to the type and alters the parameters
4. **Polymorphism**: more than one function can have the same name, distinguished by the typing or number of parameters
5. **Overloading**: refers to using the same function name more than once
6. The **header file** contains the function declarations (.h) while the **implementation file** contains the function definition (.cpp)
7. The usual convention is to delimit user-defined library file names with double quotation marks
8. In header files we put **#ifndef** X and **#define** X file identifiers so that the preprocessor only works through the code once even if the file is referenced multiple times

## Lecture 4
1. **Stream**: channel or conduit on which data is passed from senders to receivers. Data can be sent out from the program on an output stream, or received into the program on an input stream
2. Data elements must be sent to or received from a stream one at a time, i.e. in **serial fashion**
3. "**ifstream**" (input-file-stream), while "**ofstream**" (output-file-stream)
4. '**.open()**' is used for connecting stream to files, '.close()' disconnects stream, and adds end-of-file marker
5. '**.fail()**' returns True if the previous stream operation was not successful
6. '**.get()**' extracts/reads single characters
7. '**.put()**' writes/inputs single characters
8. '**.putback()**' attempts to decrease the current location in the stream by one character, making the last character extracted from the stream once again available to be extracted by input operations
9. '**.eof()**' initially set to False, but if the ifstream is positioned at the end of the file, '.get' will set the '.eof()' to True
10. Streams can be arguments to functions, but must be **reference parameters** (not value parameters)
11. **'>>'** and **'<<'** stream operators can deal with sequences of characters 
