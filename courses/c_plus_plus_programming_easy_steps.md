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

## Chapter 4
1. The **< string >** library is part of the std namespace
2. **C++ string** is much easier to work with than char arrays in C because they **automatically resize**
3. The **cin** function stops reading the input when there is a **space**
4. The **getline(cin, str)** functions is used to get a string with spaces (until \n) and store it in str
5. **cin.ignore(256, '\n')** used to discard up to 256 characters up to a newline
6. Since string is not a native type, it cannot be converted to an int or other regular data type through casting. To do so, need to import **< sstream >** library and use a **stringstream** object with the '>>' and '<<' operators
7. **string.size()** returns character length, str.clear() empties string, string.capacity() reveals memory size,
8. **string.compare()** compares the values of the ASCII equivalent and returns 0 if identical, -1 if the argument is larger than first string, otherwise 1
9. Use **'='** to assign complete strings and **string.assign()** to assign partial strings
10. Use **string.swap()** function wherever possible to avoid creating additional variables
11. **string.find("substring",index_to_start)** searches for substring in string, and it returns the index number of first occurence if successful
12. **string.find_first_of()** finds the first occurrence of any character in the string, and **string.find_first_not_of()** finds the first occurrence of a character not in the specified string. find_last_of() and find_last_not_of() begin searching at the end of the string
13. **string.insert(index, "string")** inserts a string into another string
14. **string.erase(index, num_to_remove)** removes a specified number of characters from a string
15. **string.replace(begin_erase, num_erase, "string_add")** combines insert and erase
16. **string.substr(index, num_char)** copies a substring
17. **str.at(index)** copies a character

## Chapter 5
1. the **< fstream >** library provides functions to working with files
2. For each file that is to be opened, a filestream object needs to be created, either the '**ofstream**' (output filestream) or '**ifstream**' (input filestream) object
3. '**ofstream**' is used like **cout** to output and '**ifstream**' is like **cin** that reads input
4. Syntax: **ofstream object-name ( “file-name” ) ;**
5. After writing output to a nominated file the program should run **.close()**
6. Strings can contain **\n** and **\t** for new line and tab 
7. There are different file modes which belong to the **std::ios** namespace:
* **ios::out** opens a file to write output
* **ios::in** opens a file to read input
* **ios::app** Open a file to append output at the end of any existing content
* **ios::trunc** truncates existing file (Default)
* **ios::ate** open a file without truncating and allow data to be written anywhere in the file
* **ios::binary** treat the file as binary format rather than text so the data may be stored in non-text format
8. Multiple modes can be called using '|': **ofstream object-name ( “file-name” , ios::out|ios::binary ) ;**
9. The ifstream filestream object has a get() function that can be used in a loop to read a file and assign each character to the char variable specified as its argument, or getline() if used to read strings (getline() stops reading at \n)
10. **filestream.eof()** is used to check if at the end of a file
11. The **getline()** function has a **third optional argument** that specifies a **delimiter** to stop reading a line
12. **cout and cin can be modified with functions: **
* **width()** to set stream character width
* **fill()** to indicate empty portion if the content does not fill entire stream width
* **precision()** default precision is 6 for floating point
13. **< iostream >** library provides manipulators which modify stream using << and >> operators
14. **Syntax** and **logic** errors are "compile-time" errors, while **exceptional** errors are "run-time" errors
15. **try-catch blocks** are used for exception errors
16. **catch(exception &error):** a reference to the exception can be passed to the catch block which specifies the argument to be an exception type, and a chosen exception name prefixed by the & reference operator
17. The C++ **< stdexcept >** library defines a number of exception classes, and exception type information defined in **< typeinfo > **
18. While cout function sends data to standard output, **cerr** ends error data to standard error output

## Chapter 6
1. The location in computer memory is expressed in **hexadecimal** format and can be revealed by the **& reference** operator
2. **Pointers** are variables that **store the memory address** of other variables
3. The pointers data type must match the variable to which it points to
4. A pointer variable is initialised by assigning it the memory address of another variable using the & reference operator
5. *** is the dereferencing/indirection operation** and when used on a pointer will return the variables value
6. Pointer variables can be assigned another address or changed using pointer arithmetic, useful for arrays whose elements occupy consecutive memory addresses
7. The name of an array acts like a pointer to its first element
8. When variables are passed to functions their data is **passed by value** i.e. the function operates on a copy. In contrast, when pointers are passed their data is **passed by reference** i.e. function operates on the original value
9. A pointer to a constant char array can be assigned a string of characters
10. Reference declaration syntax: **int& rNum = num ;**
11. Unlike pointers, there is no way to get the address of the **reference**, which is a true **alias** to its associated item
12. Passing by reference is more efficient than passing by value so the use of pointers and references should be encouraged
13. **Always use a reference** unless you **do not want to initialise** in the declaration or want to be **able to reassign** another variable

## Chapter 7
1. A **class** is a data structure which contains both **variables** and **functions** (methods) - both are known as members
2. Access to class members is controlled by **access specifiers** in the class declaration:
* **Public**: accessible from any place where the class is visible
* **Private**: accessible only to other members of the same class
* **Protected**: accessible only to other members of the same class and members to classes derived from that class
3. The **default** is **private**
4. It is convention to begin **class** names with an **upper case letter** and **objects** with a **lower case letter**
5. '**setter**' methods **assign** data while '**getter**' methods **retrieve** data
6. Typically the **methods** are **public** and **variables** are **private**
7. For methods with more than two lines, they should be declared as a prototype and defined separately. The definition is written as **class::function**
8. Where a class method definition has an argument of the same name as the class member, the **this -> class pointer** can be used to explictly refer to the class member variable
9. Class variables can be initialised using a constructor, and is always named exactly as the class name and requires arguments to set the initial value of class variables
10. The destructor is always named exactly as the class name but with a preceding **~**
11. Constructor methods can be overloaded
12. **Inheritance** syntax: a derived class declaration adds a colon : after its class name followed by an access specifier and class from which it derives. Commas separate class declarations derived from more than one class
13. Derived classes do not inherit the constructor or destructor, although the default constructor of the base class is always called when a new object of a derived class is created and the base class destructor is called when the object gets destroyed
14. A method can be declared in a derived class to override a matching method in the base class
15. A single overriding method in a derived class will hide all overloaded methods of that name in the base class
