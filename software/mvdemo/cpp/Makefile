CC = gcc
CPP = g++

TARGET = mvdemo

#directories
SRCDIR      := src
BUILDDIR    := obj
TARGETDIR   := bin
SRCEXT      := cpp
OBJEXT      := o

#IDIRS = /usr/include
DEFINES = -D_ISOC99_SOURCE -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_POSIX_C_SOURCE=200112 -D_XOPEN_SOURCE=600 -DPIC -DZLIB_CONST -D_GNU_SOURCE=1 -D_REENTRANT -D__STDC_CONSTANT_MACROS
CFLAGS = -fomit-frame-pointer -fPIC -pthread -Wall -Werror -Wextra -DNDEBUG -O3 -g -rdynamic $(IDIRS) $(DEFINES)
CFLAGS += -Wno-unused-function

CPPFLAGS = -std=c++11 $(CFLAGS)

#LIBRARIES = -L/usr/lib/arm-linux-gnueabihf -L/usr/local/lib -L/usr/lib
LIBRARIES = -L/usr/lib/arm-linux-gnueabihf

LFLAGS_OCV = -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_core
LFLAGS = -lmvnc -lm -lstdc++ -llzma -lz -ldl -lpthread
LDFLAGS = $(LIBRARIES) $(LFLAGS_OCV) $(LFLAGS)

#---------------------------------------------------------------------------------
#DO NOT EDIT BELOW THIS LINE
#---------------------------------------------------------------------------------
SOURCES     := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS     := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.$(OBJEXT)))

#Defauilt Make
all: directories $(TARGET)

#Remake
remake: cleaner all

#Make the Directories
directories: 
	@mkdir -p $(TARGETDIR)
	@mkdir -p $(BUILDDIR)

#Clean only Objecst
clean: 
	@$(RM) -rf $(BUILDDIR)

#Full Clean, Objects and Binaries
cleaner: clean
	@$(RM) -rf $(TARGETDIR)

#Compile
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@echo Compiling $<
	@mkdir -p $(dir $@)
	@$(CPP) $(CPPFLAGS) -c -o $@ $<

#Link
$(TARGET): $(OBJECTS)
	@echo Building $(TARGETDIR)/$(TARGET)
	@$(CPP) -o $(TARGETDIR)/$(TARGET) $^ $(LDFLAGS)

#Non-File Targets
.PHONY: all remake clean cleaner directories
