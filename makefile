CC := icc
CXX := icpc
LINK := icc

SRCDIR_BASE := src/network_executor/
BUILDDIR_BASE := build/

BIN_PATH := build/program


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
C_SOURCES += support/shared_data/implementation/shared_arrays.c
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
INCLUDE_PATH += support/shared_data/include
INCLUDE_PATH += support/testing/include




EXTLIBS := /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_intel_lp64.a
EXTLIBS += /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_intel_thread.a
EXTLIBS += /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_core.a
EXTLIBS += /opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libiomp5md.a

CFLAGS := 
CCFLAGS := -Wall -qopenmp
CPPFLAGS := -Wall -qopenmp -std=c++11
LINKFLAGS := -qopenmp -nodefaultlib:vcomp



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
	echo "build done"

rebuild: clean prebuild $(BIN_PATH)
	echo "rebuild done"

.PHONY: clean
clean: 
	echo "deleting build files"
	rm $(C_OBJECTS)
	rm $(CPP_OBJECTS)
	rm $(BIN_PATH)
	echo "done deleting"

.PHONY: prebuild
prebuild: 
	echo "building files"


$(BIN_PATH): $(C_OBJECTS) $(CPP_OBJECTS)
	$(LINK) -o $@ $(CFLAGS) $(LINKFLAGS) $(C_OBJECTS) $(CPP_OBJECTS) $(EXTLIBS)

$(C_OBJECTS): $(C_SRC_FULL) $(INC_FULLPATH) objdirs
	$(CC) -o $@ -c $(patsubst $(BUILDDIR_BASE)%.o, $(SRCDIR_BASE)%.c, $@) $(CFLAGS) $(CCFLAGS) $(INC_FLAGS)

$(CPP_OBJECTS): $(CPP_SRC_FULL) $(INC_FULLPATH) objdirs
	$(CXX) -o $@ -c $(patsubst $(BUILDDIR_BASE)%.o, $(SRCDIR_BASE)%.cpp, $@) $(CFLAGS) $(CPPFLAGS) $(INC_FLAGS)

.PHONY: objdirs
objdirs: 
	mkdir -p $(dir $(C_OBJECTS) $(CPP_OBJECTS))

