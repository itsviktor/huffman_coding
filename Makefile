main: main.c
	gcc -Wall -Wextra -o main main.c

run: main
	./main