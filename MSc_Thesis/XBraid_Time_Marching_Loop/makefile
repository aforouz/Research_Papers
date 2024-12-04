# Makefile for compiling HYPRE example code

# Define compiler and flags
EX = mpirun
CC = mpicc

HYPRE_DIR = /mnt/c/ALI/Library/hypre/src/hypre
HYPRE_FLAGS = -I $(HYPRE_DIR)/include 
HYPRE_LIB = -L $(HYPRE_DIR)/lib -lHYPRE
HYPRE_LIB_FILE = $(HYPRE_DIR)/lib/libHYPRE.a

BRAID_DIR= /mnt/c/ALI/Library/xbraid/braid
BRAID_FLAGS = -I$(BRAID_DIR)
BRAID_LIB_FILE = $(BRAID_DIR)/libbraid.a


# Define source and output files
SRC = example_serial.c
OBJ = $(SRC:.c=.o)
EXEC = example.out

# Compile the source code
Run:
	clear
	make Compile
	make Clean
#	clear
	make Report
	
Compile:
	$(CC) $(SRC) -g -o $(EXEC) -lm $(HYPRE_FLAGS) $(HYPRE_LIB) $(BRAID_FLAGS) $(BRAID_LIB_FILE)

Clean:
	rm -f $(OBJ)

Report:
	$(EX) -np 1 ./$(EXEC)

# -g -fsanitize=address