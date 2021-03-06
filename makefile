# native usage mode requires prebuilt libraries in this directory
# ${MKLROOT}/lib/mic
MK_PHI_USAGE_MODEL ?= autooffload
#MK_PHI_USAGE_MODEL ?= native

CC := icc
CXX := icpc
LINK := icc
TIME_MEASUREMENT := /usr/bin/time

SRCDIR_BASE := src/network_executor/
BUILDDIR_BASE := build/

BIN_PATH := build/program
REPORT_PATH := report.txt




C_SOURCES := application/implementation/main.c
C_SOURCES += network/weightgenerator/randgenerator/implementation/randgenerator.c
C_SOURCES += application/session/implementation/testsession.c
C_SOURCES += application/session/implementation/trainsession.c
C_SOURCES += network/network/implementation/network.c
C_SOURCES += network/layertypes/implementation/fullyconnected_layer.c
C_SOURCES += network/layertypes/implementation/convlayer.c
C_SOURCES += network/layertypes/implementation/maxpoollayer.c
C_SOURCES += support/mathematical/implementation/mathematics.c
C_SOURCES += support/mathematical/implementation/mkl_vector_prl.c
C_SOURCES += network/network/implementation/net_init.c
C_SOURCES += support/testing/implementation/testing.c

CPP_SOURCES := datasupply/csvsupplier/implementation/csvsupplier.cpp

INCLUDE_PATH := application/include
INCLUDE_PATH += application/session/include
INCLUDE_PATH += datasupply/csvsupplier/include
INCLUDE_PATH += network/layertypes/include
INCLUDE_PATH += network/network/include
INCLUDE_PATH += network/weightgenerator/include
INCLUDE_PATH += support/mathematical/include
INCLUDE_PATH += support/settings/include
INCLUDE_PATH += support/testing/include

EXTLIBS :=

ifeq ($(MK_PHI_USAGE_MODEL), native)
MKLROOT := /opt/intel/compilers_and_libraries_2017.5.239/linux/mkl
CFLAGS :=
CCFLAGS := -Wall -O3 -qopenmp -DMKL_ILP64 -I$(MKLROOT)/include -mmic
CPPFLAGS := -Wall -O3 -qopenmp -std=c++11 -DMKL_ILP64 -I$(MKLROOT)/include -mmic
LINKFLAGS := -qopenmp
MORELINKFLAGS := -L$(MKLROOT)/lib/mic -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
else
CFLAGS :=
CCFLAGS := -Wall -O3 -qopenmp -DMKL_ILP64 -I${MKLROOT}/include
CPPFLAGS := -Wall -O3 -qopenmp -std=c++11 -DMKL_ILP64 -I${MKLROOT}/include
LINKFLAGS := -qopenmp
MORELINKFLAGS := -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
endif


###

INC_FULLPATH := $(patsubst %, $(SRCDIR_BASE)%, $(INCLUDE_PATH))
INC_FLAGS := $(patsubst %, -I$(SRCDIR_BASE)%, $(INCLUDE_PATH))

C_SRC_FULL = $(patsubst %, $(SRCDIR_BASE)%, $(C_SOURCES))
CPP_SRC_FULL = $(patsubst %, $(SRCDIR_BASE)%, $(CPP_SOURCES))

C_OBJECTS := $(patsubst %.c, $(BUILDDIR_BASE)%.o, $(C_SOURCES))
CPP_OBJECTS := $(patsubst %.cpp, $(BUILDDIR_BASE)%.o, $(CPP_SOURCES))

###


.PHONY: all
all: prebuild $(BIN_PATH)
	@echo "+++ build done +++"

.PHONY: execute
execute: all preexecute $(REPORT_PATH)
	@echo "+++ execution done +++"

.PHONY: rebuild
rebuild: clean prebuild $(BIN_PATH)
	@echo "+++ rebuild done +++"

.PHONY: clean
clean:
	echo "+++ deleting build files +++"
	-rm $(C_OBJECTS)
	-rm $(CPP_OBJECTS)
	-rm $(BIN_PATH)
	echo "+++ done deleting +++"

.PHONY: prebuild
prebuild:
	@echo "+++ building files +++"

.PHONY: preexecute
preexecute:
	@echo "+++ executing program +++"



$(REPORT_PATH): $(BIN_PATH)
	$(TIME_MEASUREMENT) --verbose --append -o $@ $(BIN_PATH) > $@

$(BIN_PATH): $(C_OBJECTS) $(CPP_OBJECTS)
	$(LINK) -o $@ $(CFLAGS) $(LINKFLAGS) $(C_OBJECTS) $(CPP_OBJECTS) $(EXTLIBS) $(MORELINKFLAGS)

$(C_OBJECTS): $(C_SRC_FULL) $(INC_FULLPATH) objdirs
	$(CC) -o $@ -c $(patsubst $(BUILDDIR_BASE)%.o, $(SRCDIR_BASE)%.c, $@) $(CFLAGS) $(CCFLAGS) $(INC_FLAGS)

$(CPP_OBJECTS): $(CPP_SRC_FULL) $(INC_FULLPATH) objdirs
	$(CXX) -o $@ -c $(patsubst $(BUILDDIR_BASE)%.o, $(SRCDIR_BASE)%.cpp, $@) $(CFLAGS) $(CPPFLAGS) $(INC_FLAGS)

.PHONY: objdirs
objdirs:
	mkdir -p $(dir $(C_OBJECTS) $(CPP_OBJECTS))
