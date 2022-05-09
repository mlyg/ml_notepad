# C++ programming in easy steps

## Chapter 1
1. C++ is **object-oriented** in nature compared to C which is procedural
2. **Preprocessor directive**: makes functions within standard C++ libraries available to a program e.g. '#include iostream' and 'using namespace std'; 
3. /* and */ should only be used to comment-out sections of code
4. Traditionally, returning a zero means that the program output successively
5. **float** is correct to **6** decimal places, **double** is correct to **10** decimal places
6. For **boolean** variables, always start the name with **'is'**
7. Use **lower camel case** for variables with multiple words
8. An array can store a string if the final character is **'\0'**
9. **Vectors** are like regular arrays but their size can be changed. The vector library needs to be included at the top. Vectors are declared as 'vector data-type vector-name (size, initial-value);
10. Constant names appear in upper case, and all const declarations must be initialised
11. enum provides a handy way to create a sequence of constant integers 

## Chapter 2
1. **prefix** operators change the value **immediately**, while **postfix** operators change the value **subsequently**
2. **A-Z** characters have ASCII code values **65-90** and **a-z** characters have ASCII code values **97-122**
3. Logical operators: **&&** AND, **||** OR, **!** NOT
4. **'?'** is the **ternary** operator: ( test-expression ) ? if-true-return-this : if-false-return-this ;
5. It is preferable to test for **inequality** rather than equality
6. **sizeof** operator returns the number of bytes allocated to store a variable
7. **Casting** syntax: variable-name = static_cast < data-type > variable-name ; The alternative, old version is: variable-name = ( data-type ) variable-name ;

## Chapter 3
1. If syntax: **if ( test-expression ) { statements-to-execute-when-true }**
2. If there is only one statement to execute if true then the braces are not required
3. **Switch statement** is good for replacing if-else statements, and when the test expression only evaluates one variable
4. You can add a **default** statement after the final case that will execute statements if none match
5. It is important to follow each case statement with the **break** keyword, except the default statement
6. For loop syntax: **for ( initializer ; test-expression ; incrementer ) { statements }**
7. Do-while loop syntax: **do { statements-to-be-executed } while ( test-expression ) ;**
8. **Break** statement terminates the loop when a test condition is met, while a **continue** statement terminates the particular iteration of the loop
9. The arguments in a function prototype are known as its **'formal parameters'**
10. **Three main benefits to using functions:**
* Make program code easier to understand and maintain
* Functions can be reused by other programs
* Can divide the workload on larger projects
11. Function syntax: **return-data-type function-name ( arguments-data-type-list ) ;**
12. Use the **void** keyword if the function will return no value
13. **Function prototypes** must be declared before they can be defined
14. **Function overloading** means you can use the same function name providing their arguments differ in number, data type, or both number and data type. The compiler matches the correct function call in a process known as **'function resolution'**
15. Functions that only differ by their return data type cannot be overloaded – it’s the arguments that must differ
16. Recursive function syntax: **return-data-type function-name ( argument-list ) { statements-to-be-executed ; incrementer ; conditional-test-to-recall-or-exit ; }**
17. A recursive function generally uses **more system resources**, but it can make the code **more readable**
18. **Inline declarations** may only contain one or two statements as the compiler recreates them at each calling point
